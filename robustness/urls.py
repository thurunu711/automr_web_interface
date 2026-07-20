from django.urls import path, re_path

from . import views

app_name = "robustness"

urlpatterns = [
    path("models/", views.model_list, name="model_list"),
    path("models/<int:pk>/delete/", views.model_delete, name="model_delete"),

    path("datasets/", views.dataset_list, name="dataset_list"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:pk>/delete/", views.dataset_delete, name="dataset_delete"),

    path("runs/", views.testrun_list, name="testrun_list"),
    path("runs/new/", views.testrun_create, name="testrun_create"),
    path("runs/<int:pk>/", views.testrun_detail, name="testrun_detail"),
    path("runs/<int:pk>/delete/", views.testrun_delete, name="testrun_delete"),
    re_path(r"^runs/(?P<pk>\d+)/files/(?P<relpath>.+)$", views.testrun_download_file, name="testrun_download_file"),
    path("runs/<int:pk>/csv-chart/<path:relpath>/", views.testrun_csv_chart_data, name="testrun_csv_chart_data"),

    path("live/", views.live_feed, name="live_feed"),
    path("live/snapshot/", views.live_feed_snapshot, name="live_feed_snapshot"),
]
