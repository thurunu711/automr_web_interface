from django.contrib import messages
from django.shortcuts import redirect, render

from robustness.forms import CameraSettingsForm
from robustness.models import CameraSettings


def home(request):
    context = {
        "stats": [
            {"label": "Total Users", "value": 1240, "icon": "bi-people"},
            {"label": "Revenue", "value": "$18,400", "icon": "bi-currency-dollar"},
            {"label": "Orders", "value": 320, "icon": "bi-bag-check"},
            {"label": "Open Tickets", "value": 12, "icon": "bi-life-preserver"},
        ]
    }
    return render(request, "dashboard/home.html", context)


def reports(request):
    return render(request, "dashboard/reports.html")


def settings(request):
    camera_settings = CameraSettings.get_solo()

    if request.method == "POST":
        camera_form = CameraSettingsForm(request.POST, instance=camera_settings)
        if camera_form.is_valid():
            camera_form.save()
            messages.success(request, "Camera settings saved.")
            return redirect("dashboard:settings")
    else:
        camera_form = CameraSettingsForm(instance=camera_settings)

    return render(request, "dashboard/settings.html", {"camera_form": camera_form})
