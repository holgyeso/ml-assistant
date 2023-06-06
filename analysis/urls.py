from django.urls import path
from . import views

urlpatterns = [
    path("details", views.details, name="details"),
    path("features", views.feature_details, name="feature_details"),
    path("statistics", views.stats, name="statistics"),
    path("missing", views.drop_missing, name="missing")

]