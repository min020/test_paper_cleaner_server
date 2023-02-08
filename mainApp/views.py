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
import shutil

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
            base_path = os.path.dirname(os.path.abspath(__file__))
            load_path = base_path+'/../media/'
            save_path = base_path+'/./result/'

            for i in imglist:
                img = ImgUpload(folder_name=folder_name, files=i)
                img.save()
                img_path = load_path+str(img.files)  #저장된 이미지 경로
                h = getHash(img_path)
                img_h = ImgUpload.objects.filter(hash_code=h)   #저장된 이미지 해시코드
                if not img_h:   #db에 같은 해시코드 없는 이미지만 변환
                    if not os.path.isdir(save_path+folder_name):
                        os.mkdir(save_path+folder_name)
                    img.hash_code = h
                    enhance.de_gan(img_path, save_path+folder_name+'/'+h+'.png')
                    img.result_img_path = save_path+folder_name+'/'+h+'.png'
                    img.save()
                else:
                    if not os.path.isdir(save_path+folder_name):
                        os.mkdir(save_path+folder_name)
                    img.hash_code = 0
                    img.result_img_path = img_h[0].result_img_path
                    img.save()

            now_path = os.getcwd()
            os.chdir(save_path+folder_name)
            result_list = ImgUpload.objects.filter(folder_name=folder_name)   #db에서 변환된 이미지 경로 조회
            for i in result_list:
                shutil.copy2(i.result_img_path, save_path+folder_name)
            zip_list = os.listdir(save_path+folder_name)
            with zipfile.ZipFile(save_path+folder_name+'/'+folder_name+'.zip', 'w') as my_zip:
                for i in zip_list:
                    my_zip.write(i)
                my_zip.close()
            os.chdir(now_path)
                
            s3r = boto3.resource('s3', aws_access_key_id='AKIA2QZD4ENYR4G62TEH', aws_secret_access_key='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')        
            data = open(save_path+folder_name+'/'+folder_name+'.zip', 'rb')
            s3r.Bucket('testpapergan').put_object( Key=folder_name, Body=data, ContentType='zip')
            data.close()
            return HttpResponse(json.dumps({"status": "Success"}))
        else:
            return HttpResponse(json.dumps({"status": "Failed"}))