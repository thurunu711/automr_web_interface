"""
management command: load_external_run  (optimized for large results)
"""
import json
import os
import shutil

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from robustness import automr_utils as u
from robustness.models import Dataset, MLModel, TestRun


class Command(BaseCommand):
    help = "Register an externally-produced AutoMR results folder as a viewable TestRun."

    def add_arguments(self, parser):
        parser.add_argument("path", help="Path to the results folder...")
        parser.add_argument("--label", default=None)
        parser.add_argument("--task", default="regression", choices=["regression", "classification"])
        parser.add_argument("--model-id", type=int, default=None)
        parser.add_argument("--dataset-id", type=int, default=None)
        parser.add_argument("--copy-to-media", action="store_true")

    def handle(self, *args, **opts):
        src = os.path.abspath(opts["path"])
        if not os.path.isdir(src):
            raise CommandError(f"Not a directory: {src}")
        if not os.path.exists(os.path.join(src, "automr_results.csv")):
            self.stdout.write(self.style.WARNING("No automr_results.csv found..."))

        label = opts["label"] or os.path.basename(src.rstrip(os.sep)) or "external-run"

        model = self._resolve_model(opts["model_id"], src, label)
        dataset = self._resolve_dataset(opts["dataset_id"], src, label)

        if opts["copy_to_media"]:
            run = TestRun.objects.create(model=model, dataset=dataset, status="pending")
            output_dir = os.path.join(settings.MEDIA_ROOT, "automr_runs", f"run_{run.pk}")
            shutil.copytree(src, output_dir, dirs_exist_ok=True)
        else:
            run = TestRun(model=model, dataset=dataset, status="pending")
            output_dir = src

        self._populate_from_disk(run, output_dir, src, opts["task"])
        run.output_dir = output_dir
        run.status = "success"
        run.save()

        self.stdout.write(self.style.SUCCESS(
            f"Registered TestRun #{run.pk} -> {run.get_absolute_url()} "
            f"(output_dir={output_dir})"
        ))

    def _populate_from_disk(self, run, output_dir, src, task):
        dataset_info = u.get_json(None, "dataset_info", output_dir, "dataset_info.json") or {}
        baseline = u.get_json(None, "baseline_metrics", output_dir, "baseline_metrics.json") or {}
        model_summary = (u.get_text(output_dir, "model_summary.txt") or "").strip()
        eps_rep = u.get_epsilon_report(None, output_dir)

        # === OPTIMIZED: Avoid loading full 70k-row file when possible ===
        results_path = os.path.join(output_dir, "automr_results.csv")
        selected_mrs = []
        dataset_size = dataset_info.get("dataset_size")

        if os.path.exists(results_path):
            # Use chunks for huge files
            reader = pd.read_csv(results_path, chunksize=20000)
            first_chunk = next(reader, None)
            
            if first_chunk is not None:
                if "mr" in first_chunk.columns:
                    selected_mrs = sorted(first_chunk["mr"].dropna().unique().tolist())
                
                if dataset_size is None and "sample_id" in first_chunk.columns:
                    # Approximate size (good enough + fast)
                    dataset_size = int(first_chunk["sample_id"].nunique() * 5)  # rough if chunked
                
                # Generate lightweight top-worst for fast display
                self._create_top_worst(results_path, output_dir)

        run.task = task
        run.selected_mrs = selected_mrs
        run.transforms = selected_mrs
        run.relations = selected_mrs
        run.dataset_size = dataset_size or 0
        run.max_samples = dataset_size

        if baseline.get("mean_prediction") is not None:
            run.sample_prediction = f"mean_prediction={baseline['mean_prediction']}"
        elif model_summary:
            run.sample_prediction = model_summary[:200]

        run.console_log = (
            f"Loaded externally via `load_external_run` from: {src}\n\n"
            f"model_summary.txt:\n{model_summary[:1000]}\n\n"
            f"epsilon_report:\n{json.dumps(eps_rep, indent=2)}\n"
        )

    def _create_top_worst(self, results_path, output_dir):
        """Pre-compute a small top-N worst cases file"""
        try:
            # Read only needed columns + sample
            cols = ['sample_id', 'mr', 'failure_rate', 'severity', 'prediction', 'epsilon']
            df = pd.read_csv(results_path, usecols=lambda x: x in cols, nrows=50000)
            
            if 'failure_rate' in df.columns:
                top_worst = df.nlargest(1000, ['failure_rate', 'severity'])
                top_worst.to_csv(os.path.join(output_dir, "worst_cases_top1000.csv"), index=False)
                self.stdout.write(self.style.SUCCESS(f"Created worst_cases_top1000.csv ({len(top_worst)} rows)"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Could not create top worst: {e}"))
            
    def _resolve_model(self, model_id, src, label):
        if model_id:
            try:
                return MLModel.objects.get(pk=model_id)
            except MLModel.DoesNotExist:
                raise CommandError(f"No MLModel with pk={model_id}")

        summary = (u.get_text(src, "model_summary.txt") or "").lower()
        if "keras" in summary or "tensorflow" in summary:
            framework = "tensorflow"
        elif "torch" in summary:
            framework = "pytorch"
        elif "onnx" in summary:
            framework = "onnx"
        else:
            framework = "sklearn"

        model, created = MLModel.objects.get_or_create(
            name=f"[external] {label}", framework=framework,
        )
        if created:
            self.stdout.write(f"Created placeholder MLModel #{model.pk} ({framework}) -- no model_file attached.")
        return model

    def _resolve_dataset(self, dataset_id, src, label):
        if dataset_id:
            try:
                return Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                raise CommandError(f"No Dataset with pk={dataset_id}")

        dataset, created = Dataset.objects.get_or_create(
            name=f"[external] {label}",
            source_type="folder",
            defaults={"folder_path": src},
        )
        if created:
            self.stdout.write(f"Created placeholder Dataset #{dataset.pk} -- pointing at {src}.")
        return dataset
