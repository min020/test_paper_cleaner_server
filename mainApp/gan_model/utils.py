import numpy as np

    
def split2(dataset,h,w):
    newdataset=[]
    nsize1=256
    nsize2=256
    for ii in range(0,h,nsize1): #2048
        for iii in range(0,w,nsize2): #1536
            newdataset.append(dataset[ii:ii+nsize1,iii:iii+nsize2])   
    return np.array(newdataset)

def merge_image2(splitted_images, h,w):
    image=np.zeros(((h,w)))
    nsize1=256
    nsize2=256
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2]=splitted_images[ind]
            ind=ind+1
    return np.array(image)  