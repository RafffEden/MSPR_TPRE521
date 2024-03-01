# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import cv2
from skimage.feature import hog
import os
import hashlib
from sklearn.model_selection import train_test_split
from dotenv import dotenv_values

Data = pd.DataFrame()
config = dotenv_values(".env")
UPLOAD_PATH = config["UPLOAD_PATH"]
DATA_PATH = config["DATA_PATH"]
IMG_PATH = config["IMG_PATH"]
folders = os.listdir(IMG_PATH)


def hashage(img):
    with open(img, 'rb') as b_img :
        hash_img = hashlib.sha256(b_img.read()).digest()
    return hash_img

# Function to load and preprocess images
def load_original_dataset(folder_path):
    dt_images_finale = pd.DataFrame() 
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
        
    dt_images_finale['Path'] = img_path_list 
    dt_images_finale['Image'] = images_list
    dt_images_finale = pd.DataFrame()

    for folder in folders:
        dt_images_finale = load_original_dataset(os.path.join(IMG_PATH,folder))
        dt_images_finale['Target'] = folder
        dt_images_finale['Hash'] = dt_images_finale['Path'].apply(lambda x : hashage(x))
        dt_images_finale['Features'] = dt_images_finale['Image'].apply(lambda x : hog(x, orientations = 9, pixels_per_cell= (8,8), cells_per_block=(2,2)))

        dt_images_finale = pd.concat([dt_images_finale,dt_images_finale])
            
    # print(dt_images_finale.sample(5))

    dt_images_finale.to_csv(os.path.join(DATA_PATH,"data"),sep=";",index=False)

    X = dt_images_finale["Features"]
    Y = dt_images_finale["Target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size = 0.2)
    print(len(X_train),len(X_val),len(X_test))

    return True

def load_image(folder_path):
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
    df['Hash'] = df['Path'].apply(lambda x : hashage(x))
    df['Features'] = df['Image'].apply(lambda x : hog(x, orientations = 9, pixels_per_cell= (8,8), cells_per_block=(2,2)))
    
    df.to_csv(os.path.join(DATA_PATH,"data_no_labeled.csv"),sep=";")
    return True
    
load_image(UPLOAD_PATH)