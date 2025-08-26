# pan_card_tempering_app.py

from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np
import streamlit as st

st.set_page_config(page_title="PAN Card Tampering Detection", layout="centered")

st.title("PAN Card Tampering Detection 🔍")

# Upload original and tempered images
original_file = st.file_uploader("Upload Original PAN Card Image", type=["png","jpg","jpeg"])
tempered_file = st.file_uploader("Upload Suspected Tampered PAN Card Image", type=["png","jpg","jpeg"])

if original_file and tempered_file:
    # Open images with PIL
    original = Image.open(original_file)
    tempered = Image.open(tempered_file)
    
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    col1.image(original, caption="Original PAN Card", use_container_width=True)
    col2.image(tempered, caption="Suspected Tampered PAN Card", use_container_width=True)

    # Resize images for comparison
    o = original.resize((250,160))
    t = tempered.resize((250,160))

    # Convert to OpenCV format
    original_cv = cv2.cvtColor(np.array(o), cv2.COLOR_RGB2BGR)
    tempered_cv = cv2.cvtColor(np.array(t), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
    tempered_gray = cv2.cvtColor(tempered_cv, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM)
    (score, diff) = structural_similarity(original_gray, tempered_gray, full=True)
    diff = (diff * 255).astype("uint8")

    st.write(f"**Similarity Score:** {score:.4f}")

    # Threshold difference and find contours
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Draw rectangles around differences
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tempered_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

    st.subheader("Differences Highlighted")
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB), caption="Original with Differences", use_container_width=True)
    col2.image(cv2.cvtColor(tempered_cv, cv2.COLOR_BGR2RGB), caption="Tampered with Differences", use_container_width=True)

    st.subheader("Difference Map & Threshold")
    col1, col2 = st.columns(2)
    col1.image(diff, caption="Difference Map", use_container_width=True)
    col2.image(thresh, caption="Thresholded Differences", use_container_width=True)
