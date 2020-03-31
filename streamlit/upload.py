import streamlit as st
from PIL import Image
from classify import predict

st.title("Hotdog Classifier")

file = st.file_uploader("Upload an image...", type=["png", "jpg"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")
    st.write("Evaluating...")
    label = predict(file)
    if label[1] == "hotdog":
        st.write("✅ Hotdog")
    else:
        st.write("❌ Not hotdog")
