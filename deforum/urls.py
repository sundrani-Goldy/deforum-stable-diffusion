"""
URL configuration for deforum project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path,include
from web_ui.views.generate import Generate
from rest_framework import routers
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from web_ui.views.render import TemplateRender as render

router = routers.DefaultRouter()
router.register(r"", Generate, basename="Generate")
urlpatterns = [
    path("admin/", admin.site.urls),
    path("submit", include(router.urls)),
    path("", render.index, name="index"),
    path("text2img", render.Text2Image, name="text2img"),
    path("img2img", render.Image2Image, name="img2img"),
    path("text2video", render.Text2Video, name="text2video"),
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)