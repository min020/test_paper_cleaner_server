from django.db import models
import os
import uuid


# Create your models here.

def upload_to_func(instance, filename):
    folder_name = instance.folder_name
    file_name = uuid.uuid4().hex
    extension = os.path.splitext(filename)[-1].lower() # 확장자 추출
    return "".join(
        [folder_name, "/", file_name, extension,]
    )


class ImgUpload(models.Model):
    folder_name = models.CharField(max_length=100)
    files = models.ImageField(blank=True, upload_to=upload_to_func)
