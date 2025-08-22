import streamlit as st
from signLanguage.utils.main_utils import decodeImage
import os

# Title
st.title("SignLanguage Detection App by ITsolutions(0949567820)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
INPUT_IMAGE_PATH = "inputImage.jpg"

if uploaded_file is not None:
    # Save uploaded image
    with open(INPUT_IMAGE_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    decodeImage(uploaded_file.getvalue(), INPUT_IMAGE_PATH)

    # Run YOLO detection using existing trained model
    os.system(f"cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source ../{INPUT_IMAGE_PATH}")

    # Display output image
    output_path = "yolov5/runs/detect/exp/inputImage.jpg"
    if os.path.exists(output_path):
        st.image(output_path, caption="Predicted Image")
        st.success("Prediction Done!")
        os.system("rm -rf yolov5/runs")
    else:
        st.error("Prediction failed")