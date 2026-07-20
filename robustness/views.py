import base64
import contextlib
import io
import json
import os
import pickle
import threading
import time
import traceback as tb_module

import pandas as pd

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET

from . import automr_utils as u
from .forms import (
    CameraSettingsForm,
    DatasetFolderForm,
    DatasetUploadForm,
    LiveFeedForm,
    MLModelUploadForm,
    TestRunForm,
)
from .models import (
    CameraSettings,
    Dataset,
    DatasetImage,
    MLModel,
    TestRun,
)

PRED_TRACE_PAGE_SIZE = 200
RUNS_ROOT = os.path.join(settings.MEDIA_ROOT, "automr_runs")


# ============================================================
# Models
# ============================================================

def model_list(request):
    if request.method == "POST":
        form = MLModelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            m = form.save()
            messages.success(request, f"Loaded {m.get_framework_display()} model '{m.name}'.")
            return redirect("robustness:model_list")
    else:
        form = MLModelUploadForm()

    return render(request, "robustness/model_list.html", {
        "form": form,
        "models": MLModel.objects.all(),
    })


def model_delete(request, pk):
    m = get_object_or_404(MLModel, pk=pk)
    if request.method == "POST":
        m.model_file.delete(save=False)
        m.delete()
        messages.success(request, "Model deleted.")
    return redirect("robustness:model_list")


# ============================================================
# Datasets
# ============================================================

def dataset_list(request):
    folder_form = DatasetFolderForm(prefix="folder")
    upload_form = DatasetUploadForm(prefix="upload")

    if request.method == "POST":
        if request.POST.get("form_kind") == "folder":
            folder_form = DatasetFolderForm(request.POST, prefix="folder")
            if folder_form.is_valid():
                folder_path = folder_form.cleaned_data["folder_path"]
                if not os.path.isdir(folder_path):
                    folder_form.add_error("folder_path", f"Not a directory on the server: {folder_path}")
                else:
                    ds = folder_form.save()
                    messages.success(request, f"Dataset '{ds.name}' registered — images are scanned at run time.")
                    return redirect("robustness:dataset_list")

        elif request.POST.get("form_kind") == "upload":
            upload_form = DatasetUploadForm(request.POST, request.FILES, prefix="upload")
            if upload_form.is_valid():
                ds = Dataset.objects.create(
                    name=upload_form.cleaned_data["name"],
                    source_type="upload",
                    resize_width=upload_form.cleaned_data.get("resize_width"),
                    resize_height=upload_form.cleaned_data.get("resize_height"),
                    normalize=upload_form.cleaned_data.get("normalize", False),
                )
                for f in request.FILES.getlist("upload-images"):
                    DatasetImage.objects.create(dataset=ds, image=f)
                messages.success(request, f"Dataset '{ds.name}' created with {ds.image_count} image(s).")
                return redirect("robustness:dataset_list")

    return render(request, "robustness/dataset_list.html", {
        "folder_form": folder_form,
        "upload_form": upload_form,
        "datasets": Dataset.objects.all(),
    })


def dataset_detail(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    preview_images = []
    error = None
    if ds.source_type == "folder":
        try:
            _, paths = u.load_images_from_folder(ds.folder_path, limit=ds.max_images or 40)
            preview_images = paths[:40]
        except Exception as e:
            error = str(e)
    return render(request, "robustness/dataset_detail.html", {
        "dataset": ds,
        "preview_images": preview_images,
        "error": error,
    })


def dataset_delete(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        for di in ds.images.all():
            di.image.delete(save=False)
        ds.delete()
        messages.success(request, "Dataset deleted.")
    return redirect("robustness:dataset_list")


# ============================================================
# Test runs — the core `automr.run_full_test(...)` pipeline
# ============================================================

def testrun_create(request):
    if not MLModel.objects.exists():
        messages.warning(request, "Load a model first before configuring a run.")
        return redirect("robustness:model_list")
    if not Dataset.objects.exists():
        messages.warning(request, "Create a dataset first before configuring a run.")
        return redirect("robustness:dataset_list")

    if request.method == "POST":
        form = TestRunForm(request.POST)
        if form.is_valid():
            run = form.save(commit=False)
            run.selected_mrs = form.cleaned_data["selected_mrs"]
            run.status = "pending"
            run.save()
            _execute_run(run)
            return redirect("robustness:testrun_detail", pk=run.pk)
    else:
        form = TestRunForm()

    return render(request, "robustness/testrun_form.html", {"form": form})


def _execute_run(run: TestRun):
    """Synchronous execution of automr.run_full_test(...) — same parameters
    as run_test.py. For long-running suites this is the natural place to
    hand off to a Celery task later; kept synchronous for the MVP."""
    run.status = "running"
    run.save(update_fields=["status"])

    output_dir = run.output_dir or os.path.join(RUNS_ROOT, f"run_{run.pk}")
    os.makedirs(output_dir, exist_ok=True)
    run.output_dir = output_dir

    log_buffer = io.StringIO()
    start_time = time.time()

    try:
        from automr.api import AutoMR
    except ImportError as e:
        run.status = "failed"
        run.error_message = f"Failed to import `automr.api.AutoMR`: {e}"
        run.traceback_text = tb_module.format_exc()
        run.runtime_seconds = time.time() - start_time
        run.save()
        return

    try:
        dataset_images = u.load_dataset_images(run.dataset)
        if not dataset_images:
            raise ValueError(f"No images could be loaded for dataset '{run.dataset.name}'.")

        model_obj = u.load_model_from_path(run.model.model_file.path, run.model.framework)

        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            automr = AutoMR(
                model=model_obj,
                task=run.task,
                input_type="image",
                range_threshold=run.range_threshold,
            )
            for name in list(automr.list_transforms()):
                if name not in run.selected_mrs:
                    automr.unregister_transform(name)
            ts_dir = os.path.join(output_dir, "transformation_samples")
            os.makedirs(ts_dir, exist_ok=True)
            automr.image_saver.output_dir = ts_dir
            automr.image_saver.metadata_file = os.path.join(ts_dir, "metadata.csv")
            automr.image_saver.summary_file = os.path.join(ts_dir, "transformation_summary.csv")
            transforms = automr.list_transforms()
            relations = automr.list_relations()
            sample_pred = u.predict_single(model_obj, run.model.framework, dataset_images[0])

            df, results = automr.run_full_test(
                dataset=dataset_images,
                max_samples=int(run.max_samples),
                samples_per_mr=int(run.samples_per_mr),
                show_progress=run.show_progress,
                save=True,
                output_dir=output_dir,
                verbose=run.verbose,
                epsilon_min=float(run.epsilon_min),
                epsilon_max=float(run.epsilon_max),
                epsilon_count=int(run.epsilon_count),
            )

        runtime = time.time() - start_time

        # Persist the full results dataframe to disk so it survives past
        # this request (mirrors the 'Download automr_results.csv' button).
        if df is not None:
            try:
                df.to_csv(os.path.join(output_dir, "automr_results.csv"), index=False)
            except Exception:
                pass

        pickle_ok, pickle_err = True, ""
        try:
            pickle.dumps(automr)
        except Exception as e:
            pickle_ok, pickle_err = False, str(e)

        run.status = "success"
        run.console_log = log_buffer.getvalue()
        run.transforms = transforms
        run.relations = relations
        run.sample_prediction = str(sample_pred)[:200]
        run.dataset_size = len(dataset_images)
        run.pickle_ok = pickle_ok
        run.pickle_err = pickle_err
        run.runtime_seconds = runtime
        run.save()

    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.traceback_text = tb_module.format_exc()
        run.console_log = log_buffer.getvalue()
        run.runtime_seconds = time.time() - start_time
        run.save()


def testrun_list(request):
    return render(request, "robustness/testrun_list.html", {"runs": TestRun.objects.all()})




def _count_csv_rows(path):
    with open(path, "rb") as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def _read_csv_page(path, page, page_size):
    if not path or not os.path.exists(path):
        return None, 0
    total_rows = _count_csv_rows(path)
    if total_rows == 0:
        try:
            return pd.read_csv(path, nrows=0), 0
        except Exception:
            return None, 0
    max_page = max((total_rows - 1) // page_size + 1, 1)
    page = min(max(page, 1), max_page)
    offset = (page - 1) * page_size
    df = pd.read_csv(path, skiprows=range(1, offset + 1), nrows=page_size)
    return df, total_rows


def testrun_detail(request, pk):
    run = get_object_or_404(TestRun, pk=pk)

    if run.status != "success":
        return render(request, "robustness/testrun_detail.html", {"run": run})

    output_dir = run.output_dir
    results = None
    eps_rep = u.get_epsilon_report(results, output_dir)
    MAX_TABLE_ROWS = 500  # hard cap so the HTML page can never balloon to hundreds of MB

    def _cap(df):
        if df is not None and len(df) > MAX_TABLE_ROWS:
            print(f"[testrun_detail] capping table from {len(df)} to {MAX_TABLE_ROWS} rows for display")
            return df.head(MAX_TABLE_ROWS)
        return df

    eps_summary = _cap(u.get_df(results, "epsilon_summary", output_dir, "epsilon_summary.csv"))
    fail_summary = _cap(u.get_df(results, "failure_summary", output_dir, "failure_summary.csv"))
    sev_summary = _cap(u.get_df(results, "severity_summary", output_dir, "severity_summary.csv"))
    worst_cases = _cap(u.get_df(results, "worst_cases", output_dir, "worst_cases.csv"))
    regions = u.get_regions(results, output_dir)
    range_summary = _cap(u.get_df(results, "range_summary", output_dir, "range_summary.csv"))
    range_analysis = _cap(u.get_df(results, "range_analysis", output_dir, "range_analysis.csv"))
    baseline_metrics = u.get_json(results, "baseline_metrics", output_dir, "baseline_metrics.json")
    model_summary_text = u.get_text(output_dir, "model_summary.txt")
    dataset_info = u.get_json(results, "dataset_info", output_dir, "dataset_info.json")
    file_groups = u.categorize_generated_files(output_dir)

    # --- transformation samples: now split into sweep vs indexed ---
    transformation_samples_sweep, transformation_samples_indexed, ts_metadata = (
        u.list_transformation_samples(output_dir)
    )

    def build_entry_url(entry):
        return {
            "url": reverse("robustness:testrun_download_file", kwargs={"pk": run.pk, "relpath": entry["rel"]}),
            "name": entry["name"],
            "role": entry["role"],
            "param": entry["param"],
        }

    transformation_samples_sweep_urls = {
        mr: [build_entry_url(e) for e in entries]
        for mr, entries in transformation_samples_sweep.items()
    }
    transformation_samples_indexed_urls = {
        mr: [
            {"idx": grp["idx"], "entries": [build_entry_url(e) for e in grp["entries"]]}
            for grp in groups
        ]
        for mr, groups in transformation_samples_indexed.items()
    }

    # --- paginated prediction trace ---
    try:
        page = max(int(request.GET.get("page", 1)), 1)
    except (TypeError, ValueError):
        page = 1

    pred_trace_path = os.path.join(output_dir, "prediction_trace.csv")
    if not os.path.exists(pred_trace_path):
        pred_trace_path = os.path.join(output_dir, "automr_results.csv")

    pred_trace, pred_trace_total = _read_csv_page(pred_trace_path, page, PRED_TRACE_PAGE_SIZE)
    has_next_page = page * PRED_TRACE_PAGE_SIZE < pred_trace_total

    def to_records(df):
        return df.to_dict("records") if df is not None else None

    fail_chart = None
    if fail_summary is not None and "mr" in fail_summary.columns and "failure_rate" in fail_summary.columns:
        fail_chart = {
            "labels": fail_summary["mr"].tolist(),
            "values": [round(float(v), 4) for v in fail_summary["failure_rate"].tolist()],
        }

    sev_chart = None
    if sev_summary is not None and len(sev_summary.columns) >= 2:
        idx_col, val_col = sev_summary.columns[0], sev_summary.columns[-1]
        sev_chart = {
            "labels": sev_summary[idx_col].astype(str).tolist(),
            "values": [round(float(v), 4) for v in sev_summary[val_col].tolist()],
        }

    eps_chart = None
    if eps_summary is not None and "epsilon" in eps_summary.columns:
        y_col = "failure_rate" if "failure_rate" in eps_summary.columns else eps_summary.columns[-1]
        eps_chart = {
            "labels": [round(float(v), 4) for v in eps_summary["epsilon"].tolist()],
            "values": [round(float(v), 4) for v in eps_summary[y_col].tolist()],
        }

    range_chart = None
    if range_summary is not None and "mr" in range_summary.columns and "range_percent_change" in range_summary.columns:
        range_chart = {
            "labels": range_summary["mr"].tolist(),
            "values": [round(float(v), 6) for v in range_summary["range_percent_change"].tolist()],
        }

    worst_chart = None
    if worst_cases is not None and "severity" in worst_cases.columns:
        label_col = "mr" if "mr" in worst_cases.columns else worst_cases.columns[0]
        worst_chart = {
            "labels": [f"{v} #{i}" for i, v in enumerate(worst_cases[label_col].astype(str).tolist())],
            "values": [round(float(v), 6) for v in worst_cases["severity"].tolist()],
        }

    pred_trace_chart = None

    context = {
        "run": run,
        "media_url": settings.MEDIA_URL,
        "eps_rep": eps_rep,
        "eps_first_failure": u.find_metric(eps_rep, "First Failure Epsilon", "first_failure_epsilon"),
        "eps_recommended": u.find_metric(eps_rep, "Recommended Epsilon", "recommended_epsilon"),
        "eps_stabilization": u.find_metric(eps_rep, "Stabilization Epsilon", "stabilization_epsilon"),
        "eps_max_failure_rate": u.find_metric(eps_rep, "Maximum Failure Rate", "max_failure_rate"),
        "eps_chart": eps_chart,
        "fail_chart": fail_chart,
        "fail_summary_rows": to_records(fail_summary),
        "sev_chart": sev_chart,
        "sev_summary_rows": to_records(sev_summary),
        "worst_cases_rows": to_records(worst_cases),
        "worst_cases_cols": list(worst_cases.columns) if worst_cases is not None else None,
        "regions": regions if isinstance(regions, dict) else None,
        "regions_text": regions if isinstance(regions, str) else None,
        "range_summary_rows": to_records(range_summary),
        "range_chart": range_chart,
        "range_analysis_rows": to_records(range_analysis),
        "worst_chart": worst_chart,
        "pred_trace_chart": pred_trace_chart,
        "pred_trace": pred_trace,
        "pred_trace_page": page,
        "has_next_page": has_next_page,
        "pred_trace_total": pred_trace_total,
        "baseline_metrics": baseline_metrics,
        "model_summary_text": model_summary_text,
        "dataset_info": dataset_info,
        "file_groups": file_groups,
        "transformation_samples_sweep": transformation_samples_sweep_urls,
        "transformation_samples_indexed": transformation_samples_indexed_urls,
        "ts_metadata_rows": to_records(ts_metadata),
        "has_csv": pred_trace_total > 0,
    }
    return render(request, "robustness/testrun_detail.html", context)
def testrun_delete(request, pk):
    run = get_object_or_404(TestRun, pk=pk)
    if request.method == "POST":
        import shutil
        if run.output_dir and os.path.isdir(run.output_dir):
            shutil.rmtree(run.output_dir, ignore_errors=True)
        run.delete()
        messages.success(request, "Test run deleted.")
    return redirect("robustness:testrun_list")


def testrun_download_file(request, pk, relpath):
    run = get_object_or_404(TestRun, pk=pk)
    if not run.output_dir:
        raise Http404("No output directory for this run.")
    full_path = os.path.normpath(os.path.join(run.output_dir, relpath))
    run_dir_norm = os.path.normpath(run.output_dir)
    if not (full_path == run_dir_norm or full_path.startswith(run_dir_norm + os.sep)):
        raise Http404("Invalid path.")
    if not os.path.isfile(full_path):
        raise Http404("File not found.")
    return FileResponse(open(full_path, "rb"), as_attachment=True, filename=os.path.basename(full_path))

@require_GET
def testrun_csv_chart_data(request, pk, relpath):
    run = get_object_or_404(TestRun, pk=pk)
    if not run.output_dir:
        raise Http404("No output directory for this run.")

    full_path = os.path.normpath(os.path.join(run.output_dir, relpath))
    run_dir_norm = os.path.normpath(run.output_dir)
    if not (full_path == run_dir_norm or full_path.startswith(run_dir_norm + os.sep)):
        raise Http404("Invalid path.")
    if not os.path.isfile(full_path):
        raise Http404("File not found.")

    y_cols = request.GET.get("cols")
    y_cols = y_cols.split(",") if y_cols else None
    max_points = int(request.GET.get("max_points", 500))

    try:
        data = u.stream_aggregate_csv(full_path, y_cols=y_cols, max_points=max_points)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    if data is None:
        raise Http404("Could not read file.")

    return JsonResponse(data)
# ============================================================
# Live Feed — polls an Android "IP Webcam" phone and runs the metamorphic
# suite frame-by-frame. Django has no equivalent of Streamlit's
# st.fragment(run_every=...), so the page instead does plain JS
# setInterval() polling against `live_feed_snapshot` below, which returns
# JSON that the page uses to update charts/metrics/filmstrip in place.
# ============================================================

_LIVE_AUTOMR_CACHE = {}
_LIVE_CACHE_LOCK = threading.Lock()


def _get_cached_live_automr(model: MLModel, task, range_threshold, epsilon, selected_mrs):
    """Reuse a loaded model + configured AutoMR instance across polls
    instead of re-loading a (potentially large) model file every 1-3
    seconds — mirrors the Streamlit tab's live_automr_cache_key check."""
    from automr.api import AutoMR

    cache_key = (model.pk, model.model_file.name, task, float(range_threshold), float(epsilon), tuple(sorted(selected_mrs)))

    with _LIVE_CACHE_LOCK:
        cached = _LIVE_AUTOMR_CACHE.get(model.pk)
        if cached and cached["key"] == cache_key:
            return cached["model_obj"], cached["automr"]

        model_obj = u.load_model_from_path(model.model_file.path, model.framework)
        automr = AutoMR(model=model_obj, task=task, input_type="image",
                         epsilon=float(epsilon), range_threshold=float(range_threshold))
        for name in list(automr.list_transforms()):
            if name not in selected_mrs:
                automr.unregister_transform(name)

        _LIVE_AUTOMR_CACHE[model.pk] = {"key": cache_key, "model_obj": model_obj, "automr": automr}
        return model_obj, automr


def live_feed(request):
    if not MLModel.objects.exists():
        messages.warning(request, "Load a model first before starting a live feed test.")
        return redirect("robustness:model_list")

    camera_settings = CameraSettings.get_solo()
    form = LiveFeedForm(initial={"model": MLModel.objects.first().pk})

    return render(request, "robustness/live_feed.html", {
        "form": form,
        "camera_settings": camera_settings,
    })


@require_GET
def live_feed_snapshot(request):
    """Polled by the Live Feed page's JS every `update_interval` seconds.
    Fetches one frame from the phone, runs every selected metamorphic
    relation against it, and returns the numbers + a small JPEG thumbnail
    as JSON."""
    camera_settings = CameraSettings.get_solo()
    ip_url = request.GET.get("ip_webcam_url") or camera_settings.ip_webcam_url
    if not ip_url:
        return JsonResponse({"error": "No IP Webcam URL configured. Set one on the Settings page."}, status=400)

    try:
        model_id = int(request.GET.get("model_id"))
        model = get_object_or_404(MLModel, pk=model_id)
        task = request.GET.get("task", "regression")
        range_threshold = float(request.GET.get("range_threshold", 5.0))
        epsilon = float(request.GET.get("live_epsilon", 0.05))
        samples_per_mr = int(request.GET.get("samples_per_mr", 5))
        selected_mrs = [m for m in request.GET.get("selected_mrs", "").split(",") if m]
        frame_index = int(request.GET.get("frame_index", 0))
        resize_w = request.GET.get("resize_width", "").strip()
        resize_h = request.GET.get("resize_height", "").strip()
        resize_to = (int(resize_w), int(resize_h)) if resize_w and resize_h else None
    except (TypeError, ValueError) as e:
        return JsonResponse({"error": f"Invalid parameters: {e}"}, status=400)

    if not selected_mrs:
        return JsonResponse({"error": "No metamorphic relations selected."}, status=400)

    try:
        img = u.fetch_ip_webcam_snapshot(ip_url)
        arr = u.pil_to_array(img, resize_to=resize_to)

        model_obj, automr = _get_cached_live_automr(model, task, range_threshold, epsilon, selected_mrs)

        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            df = automr.run_all_mrs(arr, samples=samples_per_mr)

        total = len(df)
        failed = int((~df["passed"]).sum()) if total else 0
        failure_rate = failed / total if total else 0.0

        per_mr = {}
        if "mr" in df.columns and total:
            for mr_name, grp in df.groupby("mr"):
                per_mr[mr_name] = {
                    "failure_rate": round(float(1.0 - grp["passed"].mean()), 4),
                    "severity": round(float(grp["severity"].mean()), 4) if "severity" in grp.columns else None,
                }

        # small thumbnail for the filmstrip, not the full-res frame
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((160, 160))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=70)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return JsonResponse({
            "frame": frame_index,
            "timestamp": time.time(),
            "failure_rate": round(failure_rate, 4),
            "total_checks": total,
            "failed": failed,
            "per_mr": per_mr,
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "traceback": tb_module.format_exc()}, status=500)