from django import forms
from .models import *

# class FolderForm(forms.ModelForm):
#     class Meta:
#         model = Folder
#         fields = ("folder_name",)

class ImgUploadForm(forms.ModelForm):
    class Meta:
        model = ImgUpload
        fields = ("folder_name", "files",)
        # widgets = {
        #     "files": forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True})),
        # }

# ImageFormSet = forms.inlineformset_factory(Folder, ImgUpload, form=ImgUploadForm)