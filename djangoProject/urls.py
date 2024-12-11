"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from djangoApp import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    # path("admin/", admin.site.urls),
    path("API_SEND_POINTS/", views.API_SEND_POINTS),
    path("API_GET_POINTS/", views.API_GET_POINTS),
    path("API_DELETE_POINTS/", views.API_DELETE_POINTS),
    path("API_EDIT_POINTS/", views.API_EDIT_POINTS),
    path("API_IMAGE_SYNTHESIS/", views.API_IMAGE_SYNTHESIS),
    path("API_SOIL_REMOVE/", views.API_SOIL_REMOVE),
    path("API_FEATURE_SELECT/", views.API_FEATURE_SELECT),
    path('task-progress/<str:task_id>/', views.get_task_progress, name='get_task_progress'),
    path('download/<str:file_name>/', views.download_file, name='download_file'),  # 添加下载视图路径
    path("API_GET_TASKS/", views.API_GET_TASKS),
    path("API_DELETE_TASKS/", views.API_DELETE_TASKS),
    path("API_TASKS_RESULTS_DOWNLOAD/", views.API_TASKS_RESULTS_DOWNLOAD),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)