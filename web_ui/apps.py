from django.apps import AppConfig
# from .views import load_all
from django.core.cache import cache

class WebUiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web_ui"