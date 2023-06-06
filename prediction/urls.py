from django.urls import path
from . import views

urlpatterns = [
    path("feature-config", views.config_features, name="features"),
    path("unique-vals/<column_name>", views.unique_vals_in_column, name="unique_vals_in_column"),
    path("model-config", views.model_config, name="model_config"),
    path("predict", views.make_predictions, name="make_predictions")
]