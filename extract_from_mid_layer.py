import torch, torchvision

import os,sys
os.chdir("./repos/torchxrayvision/scripts")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage, skimage.filters
import pprint

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms

import torchxrayvision as xrv

from IPython.display import Image
import pandas as pd

from skimage import io

from IPython.display import clear_output

os.chdir("../../../")


def extract_features(imgs_path,features_output_path):
    
    try:
        os.chdir("./repos/torchxrayvision/scripts")
    except Exception:
        pass
    
    imgs_path = os.path.join('../../../',imgs_path)
    features_output_path = os.path.join('../../../',features_output_path)
    
    list_of_imgs = os.listdir(imgs_path)

    cuda_= torch.cuda.is_available()
    feats_=True

    feats_output = []


    count_ = 0
    for imgs_xrs in range(len(list_of_imgs)):

        img_path = os.path.join(imgs_path, list_of_imgs[imgs_xrs])

        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255) 

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]    

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                  xrv.datasets.XRayResizer(224)])
        img = transform(img)

        model = xrv.models.DenseNet(weights="all")

        output = {}
        with torch.no_grad():
          img = torch.from_numpy(img).unsqueeze(0)
          if cuda_:
              img = img.cuda()
              model = model.cuda()

    #     if feats_:
    #       img = img.requires_grad_()
        feats = model.features(img)

        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))

          #grads = torch.autograd.grad(feats[0][which_feature], img)[0][0][0]

        output["feats"] = feats.cpu().detach().numpy().reshape(-1)
        #print(list_of_imgs[imgs_xrs])

        feats_output.append(output)

        count_ +=1
        clear_output(wait = True)
        print('images processed =',count_,'/',len(list_of_imgs))
        
        
    feats_array=np.zeros((len(feats_output),1024))
    for i in range(len(feats_output)):
        feats_array[i,:] = feats_output[i]["feats"] #['feats']

    feat_dataframe = pd.DataFrame(feats_array)
    feat_dataframe.to_csv(features_output_path)

    print('shape of features:', feats_array.shape)
    print('saved to:', features_output_path)

    os.chdir("../../../")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        