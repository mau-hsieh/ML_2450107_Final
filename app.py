from flask import Flask, render_template, request
import cv2
import os
from PIL import Image
import base64
import numpy as np
import glob
import shutil
from time import sleep

app = Flask(__name__)

# 定義資料夾路徑
UPLOAD_FOLDER = 'static/original'
PROCESSED_FOLDER = 'static/processed'
CROP_FOLDER = 'static/cropped'
CASCADE_PATH = 'haar_carplate.xml'

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 上傳的圖片
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)
            
            # 處理圖片：車牌檢測、字元分割、組合
            processed_img_path = process_image(img_path)
            assembled_img_path = assemble_characters(img_path)
            
            # 將圖片轉為 base64 格式以顯示在頁面上
            original_img_base64 = convert_to_base64(img_path)
            processed_img_base64 = convert_to_base64(processed_img_path)
            assembled_img_base64 = convert_to_base64(assembled_img_path)
            
            return render_template(
                'index.html',
                original_img=original_img_base64,
                processed_img=processed_img_base64,
                assembled_img=assembled_img_base64
            )
    
    return render_template('index.html', original_img=None, processed_img=None, assembled_img=None)

def process_image(img_path):
    """進行車牌檢測並在圖片上畫框"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    signs = detector.detectMultiScale(gray, minSize=(76, 20), scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in signs:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(img_path))
    cv2.imwrite(processed_path, img)
    return processed_path

def assemble_characters(img_path):
    """分割車牌字元並組合到黑色背景上"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1[0]

    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    letterlist = []
    for box in letter_image_regions:
        x, y, w, h = box
        if 2 <= x <= 125 and 5 <= w <= 26 and 20 <= h < 40:
            letterlist.append((x, y, w, h))

    real_shape = []
    for i, box in enumerate(letterlist):
        x, y, w, h = box
        bg = thresh[y:y+h, x:x+w]
        real_shape.append(bg)

    newH, newW = thresh.shape
    space = 8
    bg = np.zeros((newH + space*2, newW + space*2, 1), np.uint8)
    bg.fill(0)

    for i, letter in enumerate(real_shape):
        h, w = letter.shape
        x, y, _, _ = letterlist[i]
        for row in range(h):
            for col in range(w):
                bg[space + y + row, space + x + col] = letter[row, col]

    _, bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY_INV)
    assembled_path = os.path.join(CROP_FOLDER, "assembled.jpg")
    cv2.imwrite(assembled_path, bg)
    return assembled_path

def convert_to_base64(img_path):
    """將圖片轉為 Base64 編碼"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
