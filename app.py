import streamlit as st
#from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
#from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
from PIL import Image
import cv2
from tensorflow import keras
from tensorflow.keras import models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


model = models.load_model('./covid_classification.h5')

st.title("Chest X-ray Classification")
st.header('')
st.write('0: Covid || 1: Normal || 2: Viral Pneumonia')
upload = st.sidebar.file_uploader(label='Upload the Image')
if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
    img = Image.open(upload)
    st.image(img,caption='Uploaded Image',width=200)

if st.sidebar.button('PREDICT'):
    st.sidebar.write("Result:")
    x = cv2.resize(opencv_image,(224,224))
    x = img_to_array(x)
    x = x.reshape(1, 224, 224, 3)
    y = model.predict(x)
    result = np.argmax(y,axis=1)
    final = result[0]
    if final == 0:
        st.header("Patient has Covid-19")
    elif final == 1:
        st.header("Patient is Normal")
    elif final == 2:
        st.header("Patient has Viral Pneunomia")
