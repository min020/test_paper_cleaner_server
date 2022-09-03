from django.shortcuts import render
import json
import os
import boto3
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .forms import *
from .gan_model import enhance
import hashlib
import zipfile

# Create your views here.

def getHash(path):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read()
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read()
    afile.close()
    return hasher.hexdigest()

@method_decorator(csrf_exempt, name="dispatch")
def model_form_upload(request):
    if request.method == "POST":
        imglist = request.FILES.getlist("files")
        folder_name = request.GET.get('f_name')
        if imglist:
            for i in imglist:
                img = ImgUpload.objects.create(folder_name=folder_name, files=i)
                img.save()

            base_path = os.path.dirname(os.path.abspath(__file__))
            load_path = base_path+'/../media/'
            save_path = base_path+'/./result/'
            folder_path = load_path+folder_name
            img_list = os.listdir(folder_path)
            os.mkdir(save_path+folder_name)
            for i in img_list:
                d = folder_path+'/'+i
                h = getHash(d)
                enhance.de_gan(d, save_path+folder_name+'/'+h+'.png')

            os.chdir(save_path+folder_name)
            result_list = os.listdir(save_path+folder_name)
            with zipfile.ZipFile(folder_name+'.zip', 'w') as my_zip:
                for i in result_list:
                    my_zip.write(i)
                my_zip.close()
            os.chdir(save_path)
                
            s3r = boto3.resource('s3', aws_access_key_id='AKIA2QZD4ENYR4G62TEH', aws_secret_access_key='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')        
            data = open(save_path+folder_name+'/'+folder_name+'.zip', 'rb')
            s3r.Bucket('testpapergan').put_object( Key=folder_name, Body=data, ContentType='zip')
            return HttpResponse(json.dumps({"status": "Success"}))
        else:
            return HttpResponse(json.dumps({"status": "Failed"}))
    

def img_process_model(request):
    if request.method == "GET":
        s3r = boto3.resource('s3', aws_access_key_id='AKIA2QZD4ENYR4G62TEH', aws_secret_access_key='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')
        base_path = os.path.dirname(os.path.abspath(__file__))
        load_path = base_path+'/../media/'
        save_path = base_path+'/./result/'
        img_list = os.listdir(load_path)
        for i, img in enumerate(img_list, start=1):
            enhance.de_gan(load_path+img, save_path+str(i)+'.png')
            data = open(save_path+str(i)+'.png', 'rb')
            s3r.Bucket('testpapergan').put_object( Key=str(i), Body=data, ContentType='png')

        return HttpResponse(json.dumps({"status": "Success"}))