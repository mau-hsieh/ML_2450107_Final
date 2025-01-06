import streamlit as st
import cv2
from PIL import Image
import numpy as np

# 加載車牌檢測模型
CASCADE_PATH = 'haar_carplate.xml'
detector = cv2.CascadeClassifier(CASCADE_PATH)

# Streamlit 標題
st.title("Car Plate Detection App")
st.write("Upload an image to detect car plates.")

# 上傳圖片
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 將上傳的圖片讀取為 NumPy 陣列
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 顯示原圖
    st.subheader("Original Image")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 灰階轉換並檢測車牌
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    signs = detector.detectMultiScale(gray, minSize=(76, 20), scaleFactor=1.1, minNeighbors=4)

    # 在圖片上畫出車牌框
    for (x, y, w, h) in signs:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 顯示處理後的圖片
    st.subheader("Processed Image with Detected Plates")
    st.image(img, caption="Detected Plates", use_column_width=True)

    # 如果沒有檢測到車牌
    if len(signs) == 0:
        st.warning("No plates detected!")
