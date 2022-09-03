# Generated by Django 4.0.4 on 2022-07-29 07:59

from django.db import migrations, models
import django.db.models.deletion
import mainApp.models


class Migration(migrations.Migration):

    dependencies = [
        ('mainApp', '0002_uploadfilemodel_folder_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Folder',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('folder_name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='ImgUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('files', models.ImageField(blank=True, upload_to=mainApp.models.upload_to_func)),
                ('folder', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='mainApp.folder')),
            ],
        ),
        migrations.DeleteModel(
            name='UploadFileModel',
        ),
    ]
