"""
Direct ports of the helper functions from the original Streamlit app.
Framework-agnostic so they're reused by every view (model loading, dataset
loading, and report reading are identical regardless of which page calls them).
"""
import glob
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import requests
from PIL import Image
import re

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def fetch_ip_webcam_snapshot(base_url: str, timeout: float = 4.0) -> Image.Image:
    """Fetch a single JPEG snapshot from an Android 'IP Webcam' app instance.
    base_url looks like 'http://192.168.1.23:8080' (no trailing slash,
    no /shot.jpg — that's appended here)."""
    base_url = base_url.rstrip("/")
    resp = requests.get(f"{base_url}/shot.jpg", timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


# ============================================================
# Model loading
# ============================================================

def load_model_from_path(file_path, framework):
    """Load a model already saved on disk (Django's FileField gives us a
    real path via .path, unlike Streamlit's in-memory UploadedFile)."""
    suffix = os.path.splitext(file_path)[1].lower()

    if framework == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(file_path)
        return model

    if framework == "pytorch":
        import torch
        obj = torch.load(file_path, map_location="cpu", weights_only=False)
        if hasattr(obj, "eval"):
            obj.eval()
        return obj

    if framework == "sklearn":
        if suffix == ".joblib":
            import joblib
            return joblib.load(file_path)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if framework == "onnx":
        return file_path  # onnxruntime takes a path directly

    raise ValueError(f"Unsupported framework: {framework}")


FRAMEWORK_LABELS = {
    "tensorflow": "TensorFlow / Keras",
    "pytorch": "PyTorch",
    "sklearn": "scikit-learn",
    "onnx": "ONNX",
}


def predict_single(model, framework, arr):
    """Best-effort single-image prediction, mirroring run_test.py's
    'Sample prediction: ...' line. Never raises — returns an error string
    instead so a framework quirk doesn't crash the page."""
    kind = FRAMEWORK_LABELS.get(framework, framework)
    try:
        if kind == "TensorFlow / Keras":
            x = np.expand_dims(arr.astype(np.float32), axis=0)
            pred = model.predict(x, verbose=0)
            return float(np.array(pred).flatten()[0])
        if kind == "PyTorch":
            import torch
            x = torch.tensor(arr.astype(np.float32)).unsqueeze(0)
            if x.ndim == 4 and x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            with torch.no_grad():
                out = model(x)
            return float(np.asarray(out).flatten()[0])
        if kind == "scikit-learn":
            pred = model.predict(arr.flatten().reshape(1, -1))
            return float(np.array(pred).flatten()[0])
        if kind == "ONNX":
            import onnxruntime as ort
            sess = ort.InferenceSession(model)
            input_name = sess.get_inputs()[0].name
            x = np.expand_dims(arr.astype(np.float32), axis=0)
            pred = sess.run(None, {input_name: x})
            return float(np.array(pred[0]).flatten()[0])
    except Exception as e:
        return f"(prediction failed: {e})"
    return None
import json  # add to your existing imports at the top if not already there

def stream_aggregate_csv(path, y_cols=None, x_col=None, chunksize=50_000,
                          max_points=500, agg="mean", cache=True):
    """
    Turns an arbitrarily large CSV into a small chart-ready payload without
    ever holding more than one chunk (or the final downsampled series) in
    memory. Reads the file set-by-set via pandas' chunked reader.
    """
    if not os.path.isfile(path):
        return None

    cache_path = f"{path}.chartcache.{max_points}.json"
    if cache and os.path.isfile(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass  # fall through and recompute

    agg_rows = []
    row_offset = 0
    resolved_y_cols = y_cols

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if resolved_y_cols is None:
            resolved_y_cols = [
                c for c in chunk.columns
                if pd.api.types.is_numeric_dtype(chunk[c])
            ][:6]

        if x_col is None:
            chunk_x = list(range(row_offset, row_offset + len(chunk)))
        else:
            chunk_x = list(chunk[x_col]) if x_col in chunk.columns else list(range(row_offset, row_offset + len(chunk)))

        bucket = max(1, len(chunk) // 20)
        for start in range(0, len(chunk), bucket):
            sub = chunk.iloc[start:start + bucket]
            row = {"x": chunk_x[start]}
            for c in resolved_y_cols:
                if c in sub.columns:
                    row[c] = float(sub[c].mean()) if agg == "mean" else float(sub[c].sum())
            agg_rows.append(row)

        row_offset += len(chunk)

    if len(agg_rows) > max_points:
        step = len(agg_rows) / max_points
        agg_rows = [agg_rows[int(i * step)] for i in range(max_points)]

    result = {
        "columns": resolved_y_cols or [],
        "rows": agg_rows,
        "total_rows_seen": row_offset,
    }

    if cache:
        try:
            with open(cache_path, "w") as f:
                json.dump(result, f)
        except Exception:
            pass

    return result

# ============================================================
# Dataset loading
# ============================================================

def pil_to_array(img: Image.Image, resize_to=None, normalize=False) -> np.ndarray:
    img = img.convert("RGB")
    if resize_to:
        img = img.resize(resize_to)
    arr = np.array(img).astype(np.float32 if normalize else np.uint8)
    if normalize:
        arr = arr / 255.0
    return arr


def load_images_from_folder(folder_path, resize_to=None, normalize=False, limit=None):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")
    paths = []
    for root, _, files in os.walk(folder_path):
        for fn in sorted(files):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, fn))
    paths = sorted(paths)
    if limit:
        paths = paths[:limit]
    images = []
    for p in paths:
        try:
            images.append(pil_to_array(Image.open(p), resize_to=resize_to, normalize=normalize))
        except Exception:
            continue
    return images, paths


def load_dataset_images(dataset):
    """Resolve a Dataset model instance into a list[np.ndarray], the same
    shape `automr.run_full_test(dataset=...)` expects."""
    resize_to = dataset.resize_to
    if dataset.source_type == "folder":
        images, _ = load_images_from_folder(
            dataset.folder_path, resize_to=resize_to, normalize=dataset.normalize,
            limit=dataset.max_images or None,
        )
        return images
    # uploaded images
    images = []
    for di in dataset.images.all():
        try:
            images.append(pil_to_array(Image.open(di.image.path), resize_to=resize_to, normalize=dataset.normalize))
        except Exception:
            continue
    return images


# ============================================================
# Report-reading helpers (dict from automr, falling back to disk files)
# ============================================================

 
def get_df(results, key, output_dir, filename, nrows=None, usecols=None, page=None, per_page=None):
    """Load a CSV as a DataFrame.
 
    Normal usage (unchanged): u.get_df(results, key, output_dir, filename)
        -> DataFrame or None.
 
    Paginated usage (opt-in, only pass these if you actually want a page):
        u.get_df(results, key, output_dir, filename, page=2, per_page=200)
        -> (DataFrame, total_row_count) or None.
    """
    if results and isinstance(results.get(key), pd.DataFrame):
        return results[key]
 
    if not output_dir:
        return None
 
    p = os.path.join(output_dir, filename)
    if not os.path.exists(p):
        return None
 
    try:
        if nrows is not None:
            return pd.read_csv(p, nrows=nrows, usecols=usecols)
 
        if page is not None and per_page is not None:
            page = max(page, 1)
            offset = (page - 1) * per_page
            skiprows = range(1, offset + 1) if offset else None
            df = pd.read_csv(p, skiprows=skiprows, nrows=per_page, usecols=usecols)
            with open(p, "rb") as f:
                total_rows = sum(1 for _ in f) - 1
            return df, max(total_rows, 0)
 
        return pd.read_csv(p, usecols=usecols)
 
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def get_text(output_dir, filename):
    if output_dir:
        p = os.path.join(output_dir, filename)
        if os.path.exists(p):
            try:
                return open(p, "r", encoding="utf-8", errors="replace").read()
            except Exception:
                return None
    return None


def get_json(results, key, output_dir, filename):
    if results and isinstance(results.get(key), dict):
        return results[key]
    text = get_text(output_dir, filename)
    if text:
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def get_epsilon_report(results, output_dir):
    rep = None
    if results and isinstance(results.get("epsilon_report"), dict):
        rep = results["epsilon_report"]
    else:
        text = get_text(output_dir, "epsilon_report.txt")
        if text:
            rep = {}
            for line in text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    rep[k.strip()] = v.strip()
    return rep or {}


def get_regions(results, output_dir):
    if results and isinstance(results.get("regions"), dict):
        return results["regions"]
    text = get_text(output_dir, "failure_regions.txt")
    return text


def find_metric(rep: dict, *candidates):
    """Case/spacing-insensitive lookup across possibly differing key names."""
    norm = {k.lower().replace(" ", "").replace("_", ""): v for k, v in rep.items()}
    for c in candidates:
        key = c.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


TEXT_PREVIEW_EXTS = (".txt", ".json", ".log", ".py", ".md", ".yaml", ".yml")
IMAGE_PREVIEW_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


def file_kind(rel_path):
    """Classify a generated report file for the 'preview' modal: csv (chart),
    text (raw text preview), image (inline preview), or other (download only)."""
    ext = os.path.splitext(rel_path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in TEXT_PREVIEW_EXTS:
        return "text"
    if ext in IMAGE_PREVIEW_EXTS:
        return "image"
    return "other"


def list_generated_files(output_dir):
    """Flat (relative_path, size_bytes) list of every file under output_dir,
    for the 'Generated report files' browser."""
    rows = []
    if not output_dir or not os.path.isdir(output_dir):
        return rows
    for root, _, files in os.walk(output_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, output_dir)
            rows.append((rel, os.path.getsize(fp)))
    return sorted(rows)





_FNAME_RE = re.compile(
    r'^(?P<relation>.+?)_(?:(?P<idx>\d{3})_)?(?P<param>-?\d+(?:\.\d+)?)_(?P<kind>original|transformed)$',
    re.IGNORECASE,
)


def list_transformation_samples(output_dir):
    ts_dir = os.path.join(output_dir, "transformation_samples")
    sweep_samples = {}
    indexed_samples = {}
    metadata_df = None
    if not os.path.isdir(ts_dir):
        return sweep_samples, indexed_samples, metadata_df

    meta_path = os.path.join(ts_dir, "metadata.csv")
    if os.path.exists(meta_path):
        try:
            metadata_df = pd.read_csv(meta_path)
        except Exception:
            metadata_df = None

    mr_dirs = sorted(
        d for d in os.listdir(ts_dir) if os.path.isdir(os.path.join(ts_dir, d))
    )

    for mr in mr_dirs:
        mr_dir = os.path.join(ts_dir, mr)
        all_files = sorted(
            f for f in os.listdir(mr_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        sweep_originals, sweep_transformed = [], []
        indexed = {}  # idx -> {"original": entry|None, "transformed": [entry, ...]}

        for fname in all_files:
            stem, _, ext = fname.rpartition(".")
            m = _FNAME_RE.match(stem)
            if not m:
                continue
            idx = m.group("idx")
            param = float(m.group("param"))
            kind = m.group("kind").lower()
            rel = os.path.relpath(os.path.join(mr_dir, fname), output_dir)

            if idx is None:
                if kind == "original":
                    sweep_originals.append((param, fname, rel))
                else:
                    sweep_transformed.append((param, fname, rel))
            else:
                idx = int(idx)
                grp = indexed.setdefault(idx, {"original": None, "transformed": []})
                entry = {"name": fname, "rel": rel, "role": kind, "param": None if kind == "original" else param}
                if kind == "original":
                    if grp["original"] is None or param < grp["original"].get("_param_raw", param):
                        entry["_param_raw"] = param
                        grp["original"] = entry
                else:
                    grp["transformed"].append(entry)

        # --- sweep tab: single photo, one original, ascending strengths ---
        sweep_entries = []
        if sweep_originals:
            param, original_fname, rel = sorted(sweep_originals, key=lambda t: t[0])[0]
            sweep_entries.append({"name": original_fname, "rel": rel, "role": "original", "param": None})
        for param, fname, rel in sorted(sweep_transformed, key=lambda t: t[0]):
            sweep_entries.append({"name": fname, "rel": rel, "role": "transformed", "param": param})
        if sweep_entries:
            sweep_samples[mr] = sweep_entries

        # --- indexed tab: grouped per dataset sample ---
        idx_groups = []
        for idx in sorted(indexed.keys()):
            grp = indexed[idx]
            entries = []
            if grp["original"]:
                orig = dict(grp["original"])
                orig.pop("_param_raw", None)
                entries.append(orig)
            entries.extend(sorted(grp["transformed"], key=lambda e: e["param"]))
            if entries:
                idx_groups.append({"idx": idx, "entries": entries})
        if idx_groups:
            indexed_samples[mr] = idx_groups

    return sweep_samples, indexed_samples, metadata_df

def categorize_generated_files(output_dir):
    """Group the flat file listing into UI tabs: top-level summary reports,
    one bucket per epsilon-sweep subfolder, and transformation-sample
    metadata CSVs. Per-relation transformation-sample .jpg files are
    deliberately excluded — they're already rendered as thumbnails
    elsewhere on the page, so listing all ~300 of them again here added
    nothing but noise."""
    rows = list_generated_files(output_dir)  # [(rel, size), ...]
    summary, ts_meta, other = [], [], []
    epsilon_groups = {}

    for rel, size in rows:
        parts = rel.replace("\\", "/").split("/")
        entry = {"rel": rel, "size": size, "kind": file_kind(rel)}

        if parts[0].startswith("epsilon_") and len(parts) > 1:
            epsilon_groups.setdefault(parts[0], []).append(entry)
        elif parts[0] == "transformation_samples":
            if len(parts) == 2:
                # metadata.csv / transformation_summary.csv only — skip
                # the per-relation subfolders (those are the .jpg files)
                ts_meta.append(entry)
        elif len(parts) == 1:
            summary.append(entry)
        else:
            other.append(entry)

    def eps_sort_key(label):
        try:
            return float(label.replace("epsilon_", ""))
        except ValueError:
            return label

    epsilon_tabs = [
        {"label": label, "files": sorted(files, key=lambda f: f["rel"])}
        for label, files in sorted(epsilon_groups.items(), key=lambda kv: eps_sort_key(kv[0]))
    ]

    return {
        "summary": sorted(summary, key=lambda f: f["rel"]),
        "epsilon_tabs": epsilon_tabs,
        "transformation_meta": sorted(ts_meta, key=lambda f: f["rel"]),
        "other": sorted(other, key=lambda f: f["rel"]),
    }