
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import cv2
from .utils import *
import os

import torch
from torch import nn
from torch.autograd.variable import Variable
from torchvision import transforms  
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class GeneratorNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(GeneratorNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class watermark(Dataset):

  def __init__(self, watermark_patches, transform=None):
    self.watermark_patches = watermark_patches
    self.transform = transform

  def __len__(self):
    return len(self.watermark_patches)

  def __getitem__(self, idx):
    watermark_image = self.watermark_patches[idx]

    watermark_image = np.asarray(watermark_image)

    watermark_image = Image.fromarray(watermark_image.astype(np.uint8))

    if self.transform:
       watermark_image = self.transform(watermark_image)

    return watermark_image

def getPatches(watermarked_image,mystride):
    watermarked_patches=[]
        
    h =  ((watermarked_image.shape [0] // 256) +1)*256 
    w =  ((watermarked_image.shape [1] // 256 ) +1)*256
    image_padding=np.ones((h,w))
    image_padding[:watermarked_image.shape[0],:watermarked_image.shape[1]]=watermarked_image
    
    for j in range (0,h-256,mystride):  #128 not 64
        for k in range (0,w-256,mystride):
            watermarked_patches.append(image_padding[j:j+256,k:k+256])
            
    return np.array(watermarked_patches)



def de_gan(deg_image_path, save_path):
    #path =  sys.argv[1]
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = GeneratorNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(base_path+'/gen_model.pt'))
    model.cuda()
    model.eval()

    input_size = (256,256,1)

    # deg_image_path = sys.argv[2]    
    img = cv2.imread(deg_image_path, cv2.IMREAD_GRAYSCALE)
    t, t_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh_np = np.zeros_like(img)
    thresh_np[img > t] = 255
    test_image = np.asarray(thresh_np)

    #test_image = np.asarray(Image.open(deg_image_path).convert('L'))

    h =  ((test_image.shape [0] // 256) +1)*256 
    w =  ((test_image.shape [1] // 256 ) +1)*256

    image_padding=np.ones((h,w))
    image_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

    test_image_p=split2(image_padding,h,w)

    test_list = watermark(test_image_p, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    predicted_list = DataLoader(test_list, batch_size=1)
    predicted_result = []
    for l in predicted_list:
        output = model(Variable(l).cuda().float())
        result = output[0].detach().cpu().squeeze()
        predicted_result.append(result)


    predicted_image=merge_image2(predicted_result,h,w)

    predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
    predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])

    '''
    if task == 'binarize':
        bin_thresh = 0.95
        predicted_image = (predicted_image[:,:]>bin_thresh)*1
    '''

    #save_path = sys.argv[3]

    plt.imsave(save_path, predicted_image,cmap='gray')



