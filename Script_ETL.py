# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import cv2
from skimage.feature import hog
import os
import hashlib

Mammiferes_df = pd.DataFrame(columns=['target', 'path', 'hash'])
Data = pd.DataFrame()
FOLDER_PATH = "Mammiferes"
folders = os.listdir(FOLDER_PATH)


# Function to load and preprocess images
def load_images(folder_path):
    df = pd.DataFrame() 
    images_list = []
    img_path_list = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img_path_list.append(img_path)
        # print(img_path_list)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (128, 64))  # Resize images to a common size
        images_list.append(resized_img)
        # print(images_list)
        
    df['Path'] = img_path_list 
    df['Image'] = images_list
    # print(df)
    return df


def hashage(img):
    with open(img, 'rb') as b_img :
        hash_img = hashlib.sha256(b_img.read()).digest()
    return hash_img
    
dt_images_finale = pd.DataFrame()

for folder in folders:
    dt_images = load_images(os.path.join(FOLDER_PATH,folder))
    dt_images['Target'] = folder
    dt_images['Hash'] = dt_images['Path'].apply(lambda x : hashage(x))
    dt_images['Features'] = dt_images['Image'].apply(lambda x : hog(x, orientations = 9, pixels_per_cell= (8,8), cells_per_block=(2,2)))

    dt_images_finale = pd.concat([dt_images_finale,dt_images])
        
print(dt_images_finale.sample(5))

dt_images_finale.to_csv("data.csv",sep=";",index=False)

