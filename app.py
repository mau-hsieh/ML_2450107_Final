from flask import Flask, render_template, request
import cv2
import os
from PIL import Image
import base64
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/original'
PROCESSED_FOLDER = 'static/processed'
CASCADE_PATH = 'haar_carplate.xml'

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 上傳的圖片
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)
            
            # 讀取圖片並進行車牌檢測
            processed_img_path = process_image(img_path)
            
            # 將圖片轉為 base64 格式以顯示在頁面上
            original_img_base64 = convert_to_base64(img_path)
            processed_img_base64 = convert_to_base64(processed_img_path)
            
            return render_template('index.html', original_img=original_img_base64, processed_img=processed_img_base64)
    
    return render_template('index.html', original_img=None, processed_img=None)

def process_image(img_path):
    # 讀取圖片
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 加載車牌檢測模型
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    signs = detector.detectMultiScale(gray, minSize=(76, 20), scaleFactor=1.1, minNeighbors=4)
    
    # 畫出車牌框
    for (x, y, w, h) in signs:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # 保存處理後的圖片
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(img_path))
    cv2.imwrite(processed_path, img)
    return processed_path

def convert_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
