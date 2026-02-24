import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet import preprocess_input

import pandas as pd
import numpy as np
import json
import cv2


MODEL_PATH = "model/skin_model.keras"
ClASS_PATH = "model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ClASS_PATH,'r') as f:
    class_names = json.load(f)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise ValueError("Camera unable to open...")


def preprocess_image(frame):
    
    image = cv2.resize(frame, (380,380))

    processed_img = preprocess_input(image)
    processed_img = np.expand_dims(processed_img, axis=0)

    return processed_img

while True:
    ret, frame = cam.read()

    

    processed_img = preprocess_image(frame)

    pred = model.predict(processed_img)
    idx = np.argmax(pred)
    score = float(np.max(pred))

    LABEL_TEXT = "Low Confidence"

    if score >= 0.6:
        LABEL_TEXT = class_names[idx]
    
    cv2.putText(frame, LABEL_TEXT, (100,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 1)

    cv2.imshow("Skin Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()