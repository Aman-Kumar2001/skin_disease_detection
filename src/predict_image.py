import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet import preprocess_input

import pandas as pd
import numpy as np
import json
import cv2
import sys

MODEL_PATH = "model/efficientnet_model.keras"
ClASS_PATH = "model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ClASS_PATH,'r') as f:
    class_names = json.load(f)


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    processed_img = preprocess_input(image)
    processed_img = np.expand_dims(processed_img, axis=0)

    return processed_img

def predict_image(image_path):
    img = preprocess_image(image_path)

    pred = model.predict(img)
    idx = np.argmax(pred)
    score = float(np.max(pred))

    return class_names[idx], score

if __name__ == "__main__":
    img_path = sys.argv[1]
    label, score = predict_image(img_path)

    print("Predicted disease: ", label)
    print("Confidence Score: ", score)

