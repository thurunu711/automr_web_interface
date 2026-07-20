# AutoMR Robustness Dashboard — Django port

Core pipeline (this MVP):

    Models  →  Datasets  →  Run Test  →  Results report

Everything that used to live in Streamlit's `st.session_state` is now a
Django model (`MLModel`, `Dataset`, `DatasetImage`, `TestRun`) so it
persists across requests/users instead of one browser tab's memory.

## Pages

| URL                     | Purpose                                             |
|-------------------------|------------------------------------------------------|
| `/robustness/models/`   | Upload a model file (.h5/.keras/.pt/.pth/.pkl/.joblib/.onnx) |
| `/robustness/datasets/` | Register a dataset — server folder path, or upload images |
| `/robustness/runs/new/` | Configure & run `automr.run_full_test(...)` (same params as run_test.py) |
| `/robustness/runs/<id>/`| The full report — console log, epsilon analysis, failure/severity charts, worst cases, regions, prediction trace, transformation sample gallery, generated-file browser |

## Important dependency note (found while porting)

`automr` unconditionally imports `torch` at import time
(`automr/core/range_tester.py`), even when you're testing a scikit-learn
or ONNX model. So `requirements.txt` needs `torch` regardless of which
framework(s) you actually plan to test:

```
django
pillow
numpy
pandas
automr
torch
tensorflow      # only if you'll test .h5/.keras models
onnxruntime     # only if you'll test .onnx models
joblib          # only if you'll load .joblib models
```

## What's intentionally NOT in this MVP yet (phase 2)

- **Live Feed** (phone "IP Webcam" polling tab) — needs either Django
  Channels/WebSockets or a JS `setInterval` + polling endpoint, since
  Django doesn't have Streamlit's `st.fragment(run_every=...)`.
- **Snapshot Test tab** (single-photo full sweep + custom-intensity
  single-relation sweep) — same report renderer will work for it, just
  needs its own capture-a-photo + run views.
- **Celery** — `_execute_run()` in `views.py` currently runs
  `automr.run_full_test(...)` synchronously in the request. For real
  datasets/epsilon sweeps this can take minutes; the natural next step
  is moving `_execute_run` into a Celery task and polling `TestRun.status`
  from the detail page (which already handles a `pending`/`running`
  state gracefully).
