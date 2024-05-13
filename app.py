import streamlit as st

from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

st.header("Object Detection App")
st.caption("Start The webcam ")
st.caption("Warning: Do not click Detect button before starting the webcam. It will result in error.")

# Load the model
model = load_model("C:\Users\Admin\Desktop\VScode project\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

# Define the class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']


# Fxn
@st.cache
def load_video(image_file):
        img = Image.open(image_file)
        return img

videopath = st.camera_input("Take a picture")
if videopath is not None:
    img = load_video(videopath )
    st.image(img, width=250)

def predict_label(image2):
    imgLoaded = load_img(image2, target_size=(255, 255))
    # Convert the image to an array
    img_array = img_to_array(imgLoaded)    #print(img_array)

    #print(img_array.shape)

    img_array = np.reshape(img_array, (1, 255, 255, 3))

    # Get the model predictions
    predictions = model.predict(img_array)
    #print("predictions:", predictions)

    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_label = class_names[class_index]
    return predicted_label
if st.button('Detect'):
    predicted_label = predict_label(videopath)
    st.write("The objects is predicted to be '{}'.".format(predicted_label))