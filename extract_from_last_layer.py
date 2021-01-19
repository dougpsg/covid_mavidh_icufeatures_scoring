#TODO
    #Fix need to go into scripts folder to load the function 


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




def extract_features(imgs_path,features_output_path,img_id_path):
    
    try:
        os.chdir("./repos/torchxrayvision/scripts")
    except Exception:
        pass
    
    features_output_path = os.path.join('../../../',features_output_path)
    imgs_path = os.path.join('../../../',imgs_path)
    img_id_path = os.path.join('../../../',img_id_path)
    
    list_of_imgs = os.listdir(imgs_path)

    # print imgs id (filename)
    file_features_names = open(img_id_path,'w')
    for file_name in list_of_imgs:
        file_features_names.write(file_name)
        file_features_names.write(',')
    file_features_names.close()
    
    
    cuda_=torch.cuda.is_available()
    feats=False

    dict_output={}

    preds_all=np.zeros((18,len(list_of_imgs)))

    count_ = 0
    for imgs_xrs in range(len(list_of_imgs)):

      img_path = imgs_path + list_of_imgs[imgs_xrs]

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

          preds = model(img).cpu()

          preds_all[:,imgs_xrs]=preds[0].detach().numpy()

          count_ +=1
          clear_output(wait = True)
          print('images processed =',count_,'/',len(list_of_imgs))

      dict_output=dict(zip(xrv.datasets.default_pathologies,preds_all))

    features1 = pd.DataFrame.from_dict(dict_output)
    features1.to_csv(features_output_path)

    print('shape of features:', features1.shape)
    print('saved to:', features_output_path)
    

    
    os.chdir("../../../")
    
    
    
class PneumoniaSeverityNet(torch.nn.Module):
    def __init__(self):
        super(PneumoniaSeverityNet, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(np.asarray((0.5484423041343689, 2.5535977)))
        
    def forward(self, x):
        preds = self.model(x)
        return preds


def extract_heatmap(imgs_path_dic, saliency_path_dic, which_feature, gaussian_filter_size):
    
    
    
    model2 = PneumoniaSeverityNet()

    try:
        os.chdir("./repos/torchxrayvision/scripts")
    except Exception:
        pass
                                  
    out_grad_images = {}
    out_grad_images['pth1']=[]
    out_grad_images['pth2']=[]

    images_dict = {}
    images_dict['pth1']=[]
    images_dict['pth2']=[]
                                  
    count1 = 1                              
    for pathx in imgs_path_dic:
    
        imgs_path=imgs_path_dic[pathx]
        saliency_path=saliency_path_dic[pathx]

        list_of_imgs = os.listdir(imgs_path)

        # delete files in the destine folder
        if len(os.listdir(saliency_path) ) != 0:
          for filename in os.listdir(saliency_path):
            file_path = os.path.join(saliency_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  

        count_ = 0
        for imgs_xrs in range(len(list_of_imgs)): #range(1):#

            img_path = imgs_path + list_of_imgs[imgs_xrs]

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

            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)


            img = img.requires_grad_()
            outputs = model2(img)
            grads = torch.autograd.grad(outputs[0][which_feature], img)[0][0][0]
            blurred = skimage.filters.gaussian(grads**2, sigma=(gaussian_filter_size, gaussian_filter_size), truncate=3.5)

            images_dict[pathx].append(img[0][0].detach())
            out_grad_images[pathx].append(np.asarray(blurred))

            count_ +=1
            clear_output(wait = True)
            print('images processed =',count_,'/',len(list_of_imgs))
            print('batch',1,'/',len(imgs_path_dic))
                                  
    for pathx in out_grad_images:
        imgs_path=imgs_path_dic[pathx]
        list_of_imgs = os.listdir(imgs_path)
                                  
        for i in range(len(out_grad_images[pathx])):
            out_image = images_dict[pathx][i]
            out_grad = np.asarray(out_grad_images[pathx][i])

            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(out_image.cpu(), cmap="gray", aspect='auto')
            ax.imshow(out_grad, alpha=0.8);

            #im_path_out = saliency_path_dic[pathx] + str(i) + '.jpg'
            im_path_out = os.path.join(saliency_path_dic[pathx], list_of_imgs[i])
            plt.savefig(im_path_out)
                                  
    os.chdir("../../../")

                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  


