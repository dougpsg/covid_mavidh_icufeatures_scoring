from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import scipy as sp
import scipy.ndimage
import gc
import glob
import shutil

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras

import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model

from IPython.display import clear_output


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# From: https://github.com/zhixuhao/unet/blob/master/data.py
def test_load_image(test_file, CLAHE_load, CLAHE_clip_load, CLAHE_grid_size_load, target_size=(256,256)):
    
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    
    # apply global equalization
    img = cv2.equalizeHist(img)
    
    if CLAHE_load:
    
        clahe = cv2.createCLAHE(clipLimit=CLAHE_clip_load, tileGridSize=(CLAHE_grid_size_load,CLAHE_grid_size_load))
        img = clahe.apply(img)
    
    img = img / 255
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)    
    
    return img

def test_generator(test_files, CLAHE_load, CLAHE_clip_load, CLAHE_grid_size_load, target_size=(256,256),):
    for test_file in test_files:
        yield test_load_image(test_file, CLAHE_load, CLAHE_clip_load, CLAHE_grid_size_load, target_size)

def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array        
        

def close_image(img):
    kernel = np.ones((15,15),np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if(area<15000):
            cv2.drawContours(closed, contours, i, (0, 0, 0), -1)
    closed = flood_fill(closed)
    ret2,th2 = cv2.threshold(closed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2
        
        
def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        # inline close mask
        img = close_image(img)
        
        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

        cv2.imwrite(result_file, img)
        
def segment(sourcepath, maskpath, destpath, CLAHE_save, CLAHE_clip_save, CLAHE_grid_size_save):   
    for filepath in glob.iglob(sourcepath):    

        img  = cv2.imread(filepath)
        
        width = 512
        height = 512
        dim = (width, height)
        ref_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
                
        # get the right mask
        name, ext_ = os.path.splitext(os.path.basename(filepath))
            
        maskfile = maskpath + "/{name}_{suff}{ext}".format(name=name, suff='predict', ext=ext_)

        mask_image = cv2.imread(maskfile)
        
        if(mask_image is None):
            
            print('no predicted mask - skipping')
                         
        else:    
            mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            mask = cv2.bitwise_and(ref_image, mask_image, mask=mask_image_gray)
            
            #
            image = mask
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            cropped = crop_image_only_outside(image,tol=0)
            ret = cv2.resize(cropped, (512, 512), interpolation = cv2.INTER_NEAREST)
            image = ret
        
            
            # auto-level
            image = cv2.equalizeHist(image)
            # inline CLACHE
            if CLAHE_save:
                clahe = cv2.createCLAHE(clipLimit=CLAHE_clip_save, tileGridSize=(CLAHE_grid_size_save,CLAHE_grid_size_save))
                image = clahe.apply(image)
            
    
            # convert to colour
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            #
            
            fnwrite = os.path.basename(filepath) 
            #cv2.imwrite(os.path.join(destpath, fnwrite),mask) 
            cv2.imwrite(os.path.join(destpath, fnwrite),image) 
            
            
def crop_image_only_outside(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def bn_act(x, act=True):
    #x = layers.BatchNormalization()(x)
    x = tfa.layers.InstanceNormalization()(x)
    if act == True:
        x = layers.Activation("relu")(x)
    return x
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv
def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = layers.Add()([conv, shortcut])
    return output
def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = layers.Add()([shortcut, res])
    return output
def upsample_concat_block(x, xskip):
    u = layers.UpSampling2D((2, 2))(x)
    c = layers.Concatenate()([u, xskip])
    return c
def ResUNet(input_size=(256,256,1)):
    f = [16, 32, 64, 128, 256, 512]
    inputs = Input(input_size)
    #inputs = keras.layers.Input((image_size, image_size, 3))
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    ## Bridge
    b0 = conv_block(e6, f[5], strides=1)
    b1 = conv_block(b0, f[5], strides=1)
    ## Decoder
    u1 = upsample_concat_block(b1, e5)
    d1 = residual_block(u1, f[5])
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    return model



def segment_filtered_images(model_path, images_path, output_masks_path, output_segmentation_path, 
                            CLAHE_load, CLAHE_clip_load, CLAHE_grid_size_load,
                            CLAHE_save, CLAHE_clip_save, CLAHE_grid_size_save):
    
    os.makedirs(output_masks_path, exist_ok=True)
    os.makedirs(output_segmentation_path, exist_ok=True)

    model = ResUNet(input_size=(512,512,1))
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, \
                      metrics=[dice_coef, 'binary_accuracy'])
                      
                      
    model.load_weights(model_path)
   

    #clean folder
    if len(os.listdir(output_masks_path) ) != 0:
      for filename in os.listdir(output_masks_path):
        file_path = os.path.join(output_masks_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  


    files = glob.glob(os.path.join(images_path,"*.*"))

    batch_size = 32
    n_images = len(files)
    n_batches = int(np.ceil(n_images/batch_size))

    for i in range(n_batches):

        batch_idx_in = i * batch_size
        batch_idx_fi = (i + 1) * batch_size

        if batch_idx_fi > n_images:
            batch_idx_fi = n_images
            
            
        clear_output(wait = True)
        print('processing batch =',i+1,'/',n_batches)
        print('processing images =',batch_idx_in,'to', batch_idx_fi)

        test_gen = test_generator(files[batch_idx_in:batch_idx_fi],  
                                  CLAHE_load, CLAHE_clip_load, CLAHE_grid_size_load,
                                  target_size=(512,512))
        results = model.predict(test_gen, len(files), verbose=1)
        save_result(output_masks_path, results, files[batch_idx_in:batch_idx_fi])

  
    if len(os.listdir(output_segmentation_path) ) != 0:
      for filename in os.listdir(output_segmentation_path):
        file_path = os.path.join(output_segmentation_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  

    segment((images_path + "/*.*"),output_masks_path,output_segmentation_path, 
             CLAHE_save, CLAHE_clip_save, CLAHE_grid_size_save)
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      

    