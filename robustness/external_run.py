"""
Shared logic for registering an externally-produced AutoMR results folder
as a viewable TestRun.

Used by:
  - the `load_external_run` management command (CLI, e.g. on your PC/HPC)
  - the "Import External Run" button on the Test Runs page (browser upload)

Both do exactly the same thing: point this at a folder that already
contains automr_results.csv / dataset_info.json / etc, and it registers
a TestRun row from it.
"""
import json
import os
import shutil

import pandas as pd
from django.conf import settings

from robustness import automr_utils as u
from robustness.models import Dataset, MLModel, TestRun


def register_external_run(src, label=None, task="regression", model_id=None,
                           dataset_id=None, copy_to_media=False):
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        raise ValueError(f"Not a directory: {src}")

    label = label or os.path.basename(src.rstrip(os.sep)) or "external-run"

    model = _resolve_model(model_id, src, label)
    dataset = _resolve_dataset(dataset_id, src, label)

    if copy_to_media:
        run = TestRun.objects.create(model=model, dataset=dataset, status="pending")
        output_dir = os.path.join(settings.MEDIA_ROOT, "automr_runs", f"run_{run.pk}")
        shutil.copytree(src, output_dir, dirs_exist_ok=True)
    else:
        run = TestRun(model=model, dataset=dataset, status="pending")
        output_dir = src

    _populate_from_disk(run, output_dir, src, task)
    run.output_dir = output_dir
    run.status = "success"
    run.save()
    return run


def _populate_from_disk(run, output_dir, src, task):
    dataset_info = u.get_json(None, "dataset_info", output_dir, "dataset_info.json") or {}
    baseline = u.get_json(None, "baseline_metrics", output_dir, "baseline_metrics.json") or {}
    model_summary = (u.get_text(output_dir, "model_summary.txt") or "").strip()
    eps_rep = u.get_epsilon_report(None, output_dir)

    results_path = os.path.join(output_dir, "automr_results.csv")
    selected_mrs = []
    dataset_size = dataset_info.get("dataset_size")

    if os.path.exists(results_path):
        reader = pd.read_csv(results_path, chunksize=20000)
        first_chunk = next(reader, None)

        if first_chunk is not None:
            if "mr" in first_chunk.columns:
                selected_mrs = sorted(first_chunk["mr"].dropna().unique().tolist())

            if dataset_size is None and "sample_id" in first_chunk.columns:
                dataset_size = int(first_chunk["sample_id"].nunique() * 5)

            _create_top_worst(results_path, output_dir)

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
        f"Imported externally from: {src}\n\n"
        f"model_summary.txt:\n{model_summary[:1000]}\n\n"
        f"epsilon_report:\n{json.dumps(eps_rep, indent=2)}\n"
    )


def _create_top_worst(results_path, output_dir):
    try:
        cols = ['sample_id', 'mr', 'failure_rate', 'severity', 'prediction', 'epsilon']
        df = pd.read_csv(results_path, usecols=lambda x: x in cols, nrows=50000)

        if 'failure_rate' in df.columns:
            top_worst = df.nlargest(1000, ['failure_rate', 'severity'])
            top_worst.to_csv(os.path.join(output_dir, "worst_cases_top1000.csv"), index=False)
    except Exception:
        pass


def _resolve_model(model_id, src, label):
    if model_id:
        return MLModel.objects.get(pk=model_id)

    summary = (u.get_text(src, "model_summary.txt") or "").lower()
    if "keras" in summary or "tensorflow" in summary:
        framework = "tensorflow"
    elif "torch" in summary:
        framework = "pytorch"
    elif "onnx" in summary:
        framework = "onnx"
    else:
        framework = "sklearn"

    model, _created = MLModel.objects.get_or_create(
        name=f" {label}", framework=framework,
    )
    return model


def _resolve_dataset(dataset_id, src, label):
    if dataset_id:
        return Dataset.objects.get(pk=dataset_id)

    dataset, _created = Dataset.objects.get_or_create(
        name=f" {label}",
        source_type="folder",
        defaults={"folder_path": src},
    )
    return dataset