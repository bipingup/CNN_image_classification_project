import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

MODEL_PATH = "mybatch_students_cnn_classifier.h5"

model = load_model(MODEL_PATH)

class_name = ["anu","bharti","deepak","manidhar","sudh"]

st.title("Student Image Classifier")
st.set_page_config(page_title="Student Image Classifier", page_icon=":camera:", layout="centered")

st.sidebar.title("upload Student Image")
st.markdown("This app classifies images of students into their respective names using a vanilla CNN model trained on a custom dataset.")

upload_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    img=Image.open(upload_file).convert("RGB")
    st.image(img,caption="Your Uploaded Image",use_column_width=True)

    image_resized=img.resize((128,128))
    img_array = image.img_to_array(image_resized)/255.0
    image_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(image_batch)

    predicted_class=class_name[np.argmax(prediction)]

    st.success(f"Predicted Student Name: {predicted_class}")
    st.subheader("below is your confidence score for all the class")
    print(prediction)
    for index,score in enumerate(prediction[0]):
        st.write(f"{class_name[index]}: {score}")