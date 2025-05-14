# -------------------
# IMPORTS
# -------------------
import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time

# -------------------
# MAIN
# -------------------
def main():
  # main page
  pass

# function to load and cache pretrained model
@st.cache_resource()
def load_model():
    path = "dffnetv2B0"
    # Model reconstruction from JSON file
    with open(path + '.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(path + '.h5')
    return model

 
# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]
    
    if predicted_prob >= 0.5:
        return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"

# generate selection of sample images 

# load model
classifier = load_model()




st.header("Detection of Real and Fake Human Faces")
st.image("1.jpg", use_column_width=True)
# upload an image
uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg'])

# when an image is uploaded, display image and run inference
if uploaded_image is not None:
  col1,col2,col3= st.columns([1,1,1])
  col2.image(uploaded_image, width=250)
  st.markdown("## Prediction:")
  if "Real" in get_prediction(classifier, uploaded_image) :
    st.success(get_prediction(classifier, uploaded_image))
  else:
    st.error(get_prediction(classifier, uploaded_image))


# -------------------
# SCRIPT/MODULE CHECKER
# -------------------
if __name__ == "__main__":
    main()










