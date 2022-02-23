from cgi import test
from difflib import restore
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time
from fruit import Fruit

st.title("Fruit Detection using TensorflowLite for microcontrollers")
st.header("Fruit Detection Example")
st.text("Upload Fruit Image for detection 🍎")
st.write("")


st.info("You can refer to [this]('https://www.kaggle.com/moltean/fruits') dataset for fruit images.")
st.write('---')
# uploaded_file = st.file_uploader("Upload fruit image", type=["jpg","png","jpeg"])

st.sidebar.header("🍇Fruit Detection using TensorflowLite for microcontrollers")
st.sidebar.write('---')
st.sidebar.subheader("A tensorflowLite model for fruit classification made to be used on microcontrollers like Arduino Nano BLE33 or ESP32 and trained on [Fruits_360](https://www.kaggle.com/moltean/fruits) dataset.")
st.sidebar.subheader(" You can check out source code of this website [here](https://github.com/ujjwal404/fruitClassificationTFLite).")
st.sidebar.write('---')
st.sidebar.markdown("Made by *[@UjjwalKadam](https://github.com/ujjwal404), [@TarushBajaj](https://github.com/Tarushfx)* and *[@Ujjwal](https://github.com/ujjwal2604)*")



def predict(image):
    test_image = image.resize((100,100))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    obj = Fruit(test_image)
    result = obj.predictLite()
    return result
  

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    class_btn = st.button("Classify")    
    
    if class_btn:
        if file_uploaded is None:
            st.write("")
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                 predictions = predict(image)
                 time.sleep(1)
                 st.success('Classified')
                 st.write(predictions)
        
main()
        
