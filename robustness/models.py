import os

from django.db import models
from django.urls import reverse

ALL_MRS = [
    "brightness", "rotation", "translation", "noise", "blur", "contrast", "composite",
    "rain", "snow", "fog", "sandstorm", "dust", "haze", "smoke",
    "visibility", "darkness", "temporal",
]

FRAMEWORK_CHOICES = [
    ("tensorflow", "TensorFlow / Keras (.h5 / .keras)"),
    ("pytorch", "PyTorch (.pt / .pth — full saved model)"),
    ("sklearn", "scikit-learn (.pkl / .joblib)"),
    ("onnx", "ONNX (.onnx)"),
]

FRAMEWORK_LABELS = dict(FRAMEWORK_CHOICES)

TASK_CHOICES = [
    ("regression", "Regression"),
    ("classification", "Classification"),
]

DATASET_SOURCE_CHOICES = [
    ("folder", "Server folder path"),
    ("upload", "Uploaded images"),
]

RUN_STATUS_CHOICES = [
    ("pending", "Pending"),
    ("running", "Running"),
    ("success", "Success"),
    ("failed", "Failed"),
]


class MLModel(models.Model):
    """A user-uploaded model file (TF/Keras, PyTorch, sklearn, or ONNX)."""

    name = models.CharField(max_length=200)
    framework = models.CharField(max_length=20, choices=FRAMEWORK_CHOICES)
    model_file = models.FileField(upload_to="models/%Y/%m/%d/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.name} ({self.get_framework_display()})"

    def get_absolute_url(self):
        return reverse("robustness:model_list")

    @property
    def filename(self):
        return os.path.basename(self.model_file.name)


class Dataset(models.Model):
    """A collection of images to run the metamorphic suite against —
    either loaded from a folder path already on the server, or from
    images uploaded through the browser."""

    name = models.CharField(max_length=200)
    source_type = models.CharField(max_length=10, choices=DATASET_SOURCE_CHOICES)
    folder_path = models.CharField(max_length=500, blank=True)
    resize_width = models.PositiveIntegerField(null=True, blank=True)
    resize_height = models.PositiveIntegerField(null=True, blank=True)
    normalize = models.BooleanField(default=False)
    max_images = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Only used for 'Server folder path' datasets — 0/blank = load all.",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.name} ({self.image_count} images)"

    def get_absolute_url(self):
        return reverse("robustness:dataset_detail", args=[self.pk])

    @property
    def image_count(self):
        return self.images.count()

    @property
    def resize_to(self):
        if self.resize_width and self.resize_height:
            return (self.resize_width, self.resize_height)
        return None


class DatasetImage(models.Model):
    """One image belonging to a Dataset (used for the 'upload' source type;
    for 'folder' datasets the images are read straight off disk each run)."""

    dataset = models.ForeignKey(Dataset, related_name="images", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="datasets/%Y/%m/%d/")

    def __str__(self):
        return os.path.basename(self.image.name)


class CameraSettings(models.Model):
    """Singleton row holding the Android 'IP Webcam' base URL used by the
    Live Feed page (mirrors the Streamlit sidebar's ip_webcam_url field).
    Saved from the main Settings page."""

    ip_webcam_url = models.CharField(
        max_length=300, blank=True,
        help_text="e.g. http://192.168.1.23:8080 (no trailing slash, no /shot.jpg)",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Camera settings"
        verbose_name_plural = "Camera settings"

    def __str__(self):
        return self.ip_webcam_url or "(not set)"

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class TestRun(models.Model):
    """One execution of `automr.run_full_test(...)` — the exact same
    parameters as run_test.py / the Streamlit 'Run Test' tab. Everything
    that used to live in st.session_state['run_output'] is persisted here."""

    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="test_runs")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="test_runs")

    # --- run_full_test params (mirrors run_test.py / sidebar section 3) ---
    task = models.CharField(max_length=20, choices=TASK_CHOICES, default="regression")
    range_threshold = models.FloatField(default=5.0)
    max_samples = models.PositiveIntegerField(default=10)
    samples_per_mr = models.PositiveIntegerField(default=5)
    show_progress = models.BooleanField(default=True)
    verbose = models.BooleanField(default=True)
    epsilon_min = models.FloatField(default=0.01)
    epsilon_max = models.FloatField(default=0.20)
    epsilon_count = models.PositiveIntegerField(default=4)
    selected_mrs = models.JSONField(default=list)

    # --- execution bookkeeping ---
    output_dir = models.CharField(max_length=500, blank=True)
    status = models.CharField(max_length=10, choices=RUN_STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    runtime_seconds = models.FloatField(null=True, blank=True)

    # --- results (populated on success) ---
    console_log = models.TextField(blank=True)
    transforms = models.JSONField(null=True, blank=True)
    relations = models.JSONField(null=True, blank=True)
    sample_prediction = models.CharField(max_length=200, blank=True)
    dataset_size = models.PositiveIntegerField(null=True, blank=True)
    pickle_ok = models.BooleanField(null=True, blank=True)
    pickle_err = models.TextField(blank=True)

    # --- results (populated on failure) ---
    error_message = models.TextField(blank=True)
    traceback_text = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Run #{self.pk} — {self.model.name} on {self.dataset.name} ({self.status})"

    def get_absolute_url(self):
        return reverse("robustness:testrun_detail", args=[self.pk])
