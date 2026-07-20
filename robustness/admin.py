from django.contrib import admin

from .models import CameraSettings, Dataset, DatasetImage, MLModel, TestRun


class DatasetImageInline(admin.TabularInline):
    model = DatasetImage
    extra = 0

@admin.register(CameraSettings)
class CameraSettingsAdmin(admin.ModelAdmin):
    list_display = ("ip_webcam_url", "updated_at")


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name", "framework", "uploaded_at")
    list_filter = ("framework",)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "source_type", "image_count", "uploaded_at")
    list_filter = ("source_type",)
    inlines = [DatasetImageInline]


@admin.register(TestRun)
class TestRunAdmin(admin.ModelAdmin):
    list_display = ("id", "model", "dataset", "status", "created_at", "runtime_seconds")
    list_filter = ("status", "task")
    readonly_fields = ("console_log", "traceback_text")
