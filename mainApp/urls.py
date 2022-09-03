from django.urls import URLPattern, path
from . import views

urlpatterns = [
    path('main', views.model_form_upload, name='upload'),
    path('process', views.img_process_model, name='process')
]
