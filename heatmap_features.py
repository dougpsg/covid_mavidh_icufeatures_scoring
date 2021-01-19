import pandas as pd
import skimage, skimage.filters
import numpy as np

import os
os.chdir("./repos/torchxrayvision/scripts")

import torch, torchvision
import torchxrayvision as xrv

from IPython.display import clear_output


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
    
def calc_energy(img):
    
    img_energy = np.sum(np.sum(np.power(img,2)))
    
    return img_energy

def calc_entropy(img):
    
    img_entropy = -np.sum(np.sum(img * np.log(img)))
    
    return img_entropy



def extract_heatmap_features(features_name, sub_features_name, imgs_path, heatmap_output_path, sigma_, truncate_):
    
    try:
        os.chdir("./repos/torchxrayvision/scripts")
    except Exception:
        pass
    
    heatmap_output_path = os.path.join('../../../',heatmap_output_path)
    imgs_path = os.path.join('../../../',imgs_path)


    model2 = PneumoniaSeverityNet()

    no_of_features = len(features_name)

    no_of_sub_features = len(sub_features_name)


    features_lv0 = np.zeros((1,(len(features_name))))

    image_entropy = np.zeros((1,(len(features_name))))
    image_energy = np.zeros((1,(len(features_name))))

    long1_entropy = np.zeros((1,(len(features_name))))
    long1_energy = np.zeros((1,(len(features_name))))

    long2_entropy = np.zeros((1,(len(features_name))))
    long2_energy = np.zeros((1,(len(features_name))))

    trans1_entropy = np.zeros((1,(len(features_name))))
    trans1_energy = np.zeros((1,(len(features_name))))

    trans2_entropy = np.zeros((1,(len(features_name))))
    trans2_energy = np.zeros((1,(len(features_name))))

    quad1_entropy = np.zeros((1,(len(features_name))))
    quad1_energy = np.zeros((1,(len(features_name))))

    quad2_entropy = np.zeros((1,(len(features_name)))) 
    quad2_energy = np.zeros((1,(len(features_name))))

    quad3_entropy = np.zeros((1,(len(features_name))))  
    quad3_energy = np.zeros((1,(len(features_name))))

    quad4_entropy = np.zeros((1,(len(features_name))))  
    quad4_energy = np.zeros((1,(len(features_name))))

    quad5_entropy = np.zeros((1,(len(features_name)))) 
    quad5_energy = np.zeros((1,(len(features_name))))

    quad6_entropy = np.zeros((1,(len(features_name))))  
    quad6_energy = np.zeros((1,(len(features_name))))





    list_of_imgs = os.listdir(imgs_path)

    feature_all = np.zeros((len(list_of_imgs), len(features_name)*(no_of_sub_features)))

    count_=0
    for imgs_xrs in range(len(list_of_imgs)):#range(1):

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

        #features_lv0 = outputs[0][:].detach().numpy()
        #features_lv0 = np.reshape(features_lv0,(1,no_of_features))

        for featurex in range(no_of_features):#range(1):
            grads = torch.autograd.grad(outputs[0][featurex], img,retain_graph=True)[0][0][0]
            #blurred = skimage.filters.gaussian(grads**2, sigma=(5, 5), truncate=3.5)
            blurred = skimage.filters.gaussian(grads**2, sigma=(sigma_, sigma_), truncate=truncate_)
            #blurred = grads**2

            #test_image = blurred / np.max(blurred)
            test_image = np.asarray(blurred) 

            #Whole image - energy and       
            image_entropy[0,featurex] = calc_entropy(test_image)
            image_energy[0,featurex] = calc_energy(test_image)

            #Cuts - Longitudinal, transversal, quadrants
            #Longitudinal
            long1_cord = (0,test_image.shape[0],0,int(np.floor(test_image.shape[1]/2)))
            long2_cord = (0,test_image.shape[0],int(np.floor(test_image.shape[1]/2))+1,test_image.shape[1])

            long1_slice = test_image[long1_cord[0]:long1_cord[1], long1_cord[2]:long1_cord[3]]
            long2_slice = test_image[long2_cord[0]:long2_cord[1], long2_cord[2]:long2_cord[3]]

            long1_entropy[0,featurex] = calc_entropy(long1_slice)
            long1_energy[0,featurex] = calc_energy(long1_slice)

            long2_entropy[0,featurex] = calc_entropy(long2_slice)  
            long2_energy[0,featurex] = calc_energy(long2_slice)


            #Transversal
            trans1_cord = (0,int(np.floor(test_image.shape[0]/2)),0,test_image.shape[1])
            trans2_cord = (int(np.floor(test_image.shape[1]/2))+1,test_image.shape[0],0,test_image.shape[1])

            trans1_slice = test_image[trans1_cord[0]:trans1_cord[1], trans1_cord[2]:trans1_cord[3]]     
            trans2_slice = test_image[trans2_cord[0]:trans2_cord[1], trans2_cord[2]:trans2_cord[3]]

            trans1_entropy[0,featurex] = calc_entropy(trans1_slice)
            trans1_energy[0,featurex] = calc_energy(trans1_slice)

            trans2_entropy[0,featurex] = calc_entropy(trans2_slice)  
            trans2_energy[0,featurex] = calc_energy(trans2_slice)


            #Quads
            quad1_cord = (0,int(np.floor(test_image.shape[0]/3)),0,int(np.floor(test_image.shape[1]/2)))
            quad2_cord = (0,int(np.floor(test_image.shape[0]*1/3)),int(np.floor(test_image.shape[1]/2))+1,test_image.shape[1])
            quad3_cord = (int(np.floor(test_image.shape[0]/3))+1,int(np.floor(test_image.shape[0]*2/3)),0,int(np.floor(test_image.shape[1]/2)))
            quad4_cord = (int(np.floor(test_image.shape[0]/3)),int(np.floor(test_image.shape[0]*2/3)),int(np.floor(test_image.shape[1]/2))+1,test_image.shape[1])
            quad5_cord = (int(np.floor(test_image.shape[0]*2/3))+1,test_image.shape[0],0,int(np.floor(test_image.shape[1]/2)))
            quad6_cord = (int(np.floor(test_image.shape[0]*2/3))+1,test_image.shape[0],int(np.floor(test_image.shape[1]/2))+1,test_image.shape[1])

            quad1_slice = test_image[quad1_cord[0]:quad1_cord[1], quad1_cord[2]:quad1_cord[3]]
            quad2_slice = test_image[quad2_cord[0]:quad2_cord[1], quad2_cord[2]:quad2_cord[3]]        
            quad3_slice = test_image[quad3_cord[0]:quad3_cord[1], quad3_cord[2]:quad3_cord[3]]        
            quad4_slice = test_image[quad4_cord[0]:quad4_cord[1], quad4_cord[2]:quad4_cord[3]]        
            quad5_slice = test_image[quad5_cord[0]:quad5_cord[1], quad5_cord[2]:quad5_cord[3]]        
            quad6_slice = test_image[quad6_cord[0]:quad6_cord[1], quad6_cord[2]:quad6_cord[3]]

            quad1_entropy[0,featurex] = calc_entropy(quad1_slice)
            quad1_energy[0,featurex] = calc_energy(quad1_slice)

            quad2_entropy[0,featurex] = calc_entropy(quad2_slice)  
            quad2_energy[0,featurex] = calc_energy(quad2_slice)

            quad3_entropy[0,featurex] = calc_entropy(quad3_slice)  
            quad3_energy[0,featurex] = calc_energy(quad3_slice)

            quad4_entropy[0,featurex] = calc_entropy(quad4_slice)  
            quad4_energy[0,featurex] = calc_energy(quad4_slice)

            quad5_entropy[0,featurex] = calc_entropy(quad5_slice)  
            quad5_energy[0,featurex] = calc_energy(quad5_slice)

            quad6_entropy[0,featurex] = calc_entropy(quad6_slice)  
            quad6_energy[0,featurex] = calc_energy(quad6_slice)

        #TO DO: figure a way to not hardcode this
    #     features_concat = np.concatenate((features_lv0, image_energy, long1_energy, long2_energy, trans1_energy, trans2_energy, quad1_energy, 
    #                 quad2_energy, quad3_energy, quad4_energy, quad5_energy, quad6_energy,
    #                image_entropy, long1_entropy, long2_entropy, trans1_entropy, trans2_entropy, quad1_entropy,
    #                 quad2_entropy, quad3_entropy, quad4_entropy, quad5_entropy, quad6_entropy
    #                ), axis=1)

        features_concat = np.concatenate((image_entropy, image_energy, long1_entropy,
                                         long1_energy, long2_entropy, long2_energy, trans1_entropy,
                                         trans1_energy, trans2_entropy, trans2_energy, quad1_entropy,
                                         quad1_energy, quad2_entropy, quad2_energy, quad3_entropy,
                                         quad3_energy, quad4_entropy, quad4_energy, quad5_entropy,
                                         quad5_energy, quad6_entropy, quad6_energy), axis=1)

        #Deal with nan
        idxs = np.argwhere(np.isnan(features_concat))
        for idx in idxs:
            features_concat[idx[0],idx[1]] = 0
        #print(np.argwhere(np.isnan(features_concat)))

        feature_all[imgs_xrs,:]=features_concat

        count_ +=1
        clear_output(wait = True)
        print('images processed =',count_,'/',len(list_of_imgs))

        
    
    
    heat_map_features = pd.DataFrame(feature_all)
    print('heatmap features shape:', heat_map_features.shape)
    heat_map_features.to_csv(heatmap_output_path)
    
    os.chdir("../../../")