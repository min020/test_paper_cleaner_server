from django import forms
from .models import *


class ImgUploadForm(forms.ModelForm):
    class Meta:
        model = ImgUpload
        fields = ("folder_name", "files",)