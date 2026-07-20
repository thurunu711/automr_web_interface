from django import forms

from .models import ALL_MRS, CameraSettings, Dataset, MLModel, TASK_CHOICES, TestRun

MODEL_FILE_EXTS = [".h5", ".keras", ".pt", ".pth", ".pkl", ".joblib", ".onnx"]


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """Django's FileField only ever hands back one file; this lets the
    'Upload images' dataset form accept many at once, mirroring Streamlit's
    `accept_multiple_files=True`."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"multiple": True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            return [single_file_clean(d, initial) for d in data]
        return single_file_clean(data, initial)


class MLModelUploadForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ["name", "framework", "model_file"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. traffic-sign-cnn-v2"}),
            "framework": forms.Select(attrs={"class": "form-select"}),
            "model_file": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }

    def clean(self):
        cleaned = super().clean()
        f = cleaned.get("model_file")
        if f is not None:
            import os
            ext = os.path.splitext(f.name)[1].lower()
            if ext not in MODEL_FILE_EXTS:
                raise forms.ValidationError(
                    f"Unrecognized file extension '{ext}'. Expected one of: {', '.join(MODEL_FILE_EXTS)}"
                )
        return cleaned


class DatasetFolderForm(forms.ModelForm):
    """Source A: a folder path already on the server disk (mirrors the
    Streamlit 'Folder path' sidebar option)."""

    class Meta:
        model = Dataset
        fields = ["name", "folder_path", "max_images", "resize_width", "resize_height", "normalize"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. traffic-signs-train"}),
            "folder_path": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "/data/datasets/traffic_signs/train/images",
            }),
            "max_images": forms.NumberInput(attrs={"class": "form-control", "placeholder": "0 = all"}),
            "resize_width": forms.NumberInput(attrs={"class": "form-control"}),
            "resize_height": forms.NumberInput(attrs={"class": "form-control"}),
            "normalize": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source_type = "folder"
        if commit:
            instance.save()
        return instance


class DatasetUploadForm(forms.Form):
    """Source B: images uploaded through the browser (mirrors Streamlit's
    'Upload images' file_uploader with accept_multiple_files=True)."""

    name = forms.CharField(
        max_length=200, widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. spot-check-batch-1"})
    )
    images = MultipleFileField(
        widget=MultipleFileInput(attrs={"class": "form-control", "accept": "image/png,image/jpeg,image/jpg"}),
    )
    resize_width = forms.IntegerField(required=False, widget=forms.NumberInput(attrs={"class": "form-control"}))
    resize_height = forms.IntegerField(required=False, widget=forms.NumberInput(attrs={"class": "form-control"}))
    normalize = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))


class TestRunForm(forms.ModelForm):
    """Mirrors run_test.py / the Streamlit sidebar sections 1-3 + 'Run Test' tab:
    pick a model + dataset, then the exact run_full_test(...) parameters."""

    selected_mrs = forms.MultipleChoiceField(
        choices=[(m, m) for m in ALL_MRS],
        initial=ALL_MRS,
        widget=forms.CheckboxSelectMultiple,
        label="Metamorphic relations to run",
    )

    class Meta:
        model = TestRun
        fields = [
            "model", "dataset", "task", "range_threshold",
            "max_samples", "samples_per_mr", "show_progress", "verbose",
            "epsilon_min", "epsilon_max", "epsilon_count", "output_dir",
        ]
        widgets = {
            "model": forms.Select(attrs={"class": "form-select"}),
            "dataset": forms.Select(attrs={"class": "form-select"}),
            "task": forms.Select(attrs={"class": "form-select"}),
            "range_threshold": forms.NumberInput(attrs={"class": "form-control", "step": "0.5"}),
            "max_samples": forms.NumberInput(attrs={"class": "form-control"}),
            "samples_per_mr": forms.NumberInput(attrs={"class": "form-control"}),
            "show_progress": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "verbose": forms.CheckboxInput(attrs={"class": "form-check-input"}),
            "epsilon_min": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "epsilon_max": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "epsilon_count": forms.NumberInput(attrs={"class": "form-control"}),
            "output_dir": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Leave blank for an automatic per-run folder",
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model"].queryset = MLModel.objects.all()
        self.fields["dataset"].queryset = Dataset.objects.all()
        self.fields["output_dir"].required = False
        # NOTE: no `.initial = "results"` here anymore. That line used to
        # pre-fill every new run's output_dir with the literal string
        # "results", so every run wrote to (and overwrote) the same shared
        # folder instead of getting its own. Leaving this field genuinely
        # blank lets _execute_run's fallback (RUNS_ROOT/run_<pk>) generate
        # a unique folder per run, which was always the intended behavior.

    def clean_selected_mrs(self):
        return self.cleaned_data["selected_mrs"]


class CameraSettingsForm(forms.ModelForm):
    """Shown on the main Settings page — just the IP Webcam base URL."""

    class Meta:
        model = CameraSettings
        fields = ["ip_webcam_url"]
        widgets = {
            "ip_webcam_url": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "http://192.168.1.23:8080",
            }),
        }


class LiveFeedForm(forms.Form):
    """Configures one Live Feed session — mirrors sidebar sections 1 + 3
    plus the Live Feed tab's own epsilon slider/update interval."""

    model = forms.ModelChoiceField(
        queryset=MLModel.objects.all(),
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    task = forms.ChoiceField(choices=TASK_CHOICES, initial="regression",
                              widget=forms.Select(attrs={"class": "form-select"}))
    range_threshold = forms.FloatField(initial=5.0, widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.5"}))
    samples_per_mr = forms.IntegerField(initial=5, min_value=1, widget=forms.NumberInput(attrs={"class": "form-control"}))
    live_epsilon = forms.FloatField(
        initial=0.05, min_value=0.01, max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
        label="epsilon (per-frame intensity)",
    )
    resize_width = forms.IntegerField(
        required=False, initial=64,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
        help_text="Must match what the model was trained on. Leave blank to use the raw camera resolution.",
    )
    resize_height = forms.IntegerField(
        required=False, initial=64,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    update_interval = forms.ChoiceField(
        choices=[("0.5", "0.5s"), ("1.0", "1s"), ("1.5", "1.5s"), ("2.0", "2s"), ("3.0", "3s")],
        initial="1.0",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    selected_mrs = forms.MultipleChoiceField(
        choices=[(m, m) for m in ALL_MRS],
        initial=ALL_MRS,
        widget=forms.CheckboxSelectMultiple,
        label="Metamorphic relations to run",
    )