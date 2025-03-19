import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def process_images(image1_path, image2_path):
    # Load images using the provided file paths
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        st.error("Error: One or both images could not be loaded. Check file paths.")
        return None

    # Resize image2 to match image1's dimensions
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Apply threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of detected objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    detected_image = image1.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green box

    return detected_image

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# Streamlit UI
st.title("Object Detection UI")
st.write("Upload two images to compare and highlight differences.")

# File Uploaders
image1_file = st.file_uploader("Upload Image 1 (Main Scene)", type=["jpg", "png", "jpeg"])
image2_file = st.file_uploader("Upload Image 2 (Object to Find)", type=["jpg", "png", "jpeg"])

if image1_file and image2_file:
    # Save uploaded images temporarily
    image1_path = save_uploaded_file(image1_file)
    image2_path = save_uploaded_file(image2_file)

    # Display uploaded images
    st.image([image1_file, image2_file], caption=["Image 1 - Main Scene", "Image 2 - Object to Find"], width=300)

    # Process images
    detected_image = process_images(image1_path, image2_path)

    if detected_image is not None:
        st.image(detected_image, caption="Detected Objects in Image 1", channels="BGR", use_column_width=True)
        
        # Save and provide download button
        output_path = "detected_objects.png"
        cv2.imwrite(output_path, detected_image)
        with open(output_path, "rb") as file:
            st.download_button(label="Download Processed Image", data=file, file_name="detected_objects.png", mime="image/png")
