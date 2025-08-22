import streamlit as st
import os
from signLanguage.utils.main_utils import decodeImage

st.title("SignLanguage Detection App with Live Camera")

# Camera input
camera_image = st.camera_input("Capture your image")

# Path to temporarily save the captured frame
LIVE_IMAGE_PATH = "live_input.jpg"

if camera_image is not None:
    # Save the captured frame
    with open(LIVE_IMAGE_PATH, "wb") as f:
        f.write(camera_image.getbuffer())
    
    decodeImage(camera_image.getvalue(), LIVE_IMAGE_PATH)

    # Run YOLO detection on live image
    os.system(f"cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source ../{LIVE_IMAGE_PATH}")

    # Display output image
    output_path = "yolov5/runs/detect/exp/live_input.jpg"
    if os.path.exists(output_path):
        st.image(output_path, caption="Predicted Image from Live Camera")
        os.system("rm -rf yolov5/runs")
    else:
        st.error("Prediction failed")