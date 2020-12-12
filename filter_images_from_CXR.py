import pandas as pd
import shutil
import os
import numpy as np

def filter_(conditions, metadata_path, imageDir, outputDir, save_imgs, del_files):
    
    metadata_csv = pd.read_csv(metadata_path)
    
    os.makedirs(outputDir, exist_ok=True)
    
    if del_files:
        if len(os.listdir(outputDir) ) != 0:
            for filename in os.listdir(outputDir):
                file_path = os.path.join(outputDir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
                    
    df = metadata_csv
    for condition, filt_str in conditions.items():
        if filt_str[1] == 'match':
            df=df[df[condition]==(filt_str[0])]
        elif filt_str[1] == 'exclude':
            df=df[df[condition]!=(filt_str[0])]
        
    imgs_filenames = []
    for (i, row) in df.iterrows():    
        imgs_filenames.append(row["filename"])

    if save_imgs:
        for imgs_filename in imgs_filenames:
            filePath = os.path.sep.join([imageDir, imgs_filename])
            filename, fileext = os.path.splitext(os.path.basename(imgs_filename))

            filePath_dest = os.path.sep.join([outputDir, imgs_filename])

            shutil.copy(filePath, filePath_dest)

    print('No of images =', len(imgs_filenames))
    print('images save at:', outputDir)
    