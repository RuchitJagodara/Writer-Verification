import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


TARGET_SIZE = (128, 128)
THRESHOLD = 0.5
df = pd.read_csv("/Users/Ruchit/Desktop/ML_Project/dataset/val.csv")
size = df.shape[0]
for i in range(size):
    n1, n2 = df.iloc[i, 0], df.iloc[i, 1]

    val_folder_path = os.path.join(os.getcwd(), "dataset/val")


    img1_path = os.path.join(val_folder_path, n1)
    img2_path = os.path.join(val_folder_path, n2)


    img1 = load_img(img1_path, target_size=TARGET_SIZE)
    img1 = img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_input(img1)


    img2 = load_img(img2_path, target_size=TARGET_SIZE)
    img2 = img_to_array(img2)
    img2 = np.expand_dims(img2, axis=0)
    img2 = preprocess_input(img2)
    
    loaded_model = keras.models.load_model("siamese_model.h5")
    prediction = loaded_model.predict([img1, img2])
    binary_prediction = 1 if prediction > THRESHOLD else 0
    df.iloc[i, 2] = binary_prediction