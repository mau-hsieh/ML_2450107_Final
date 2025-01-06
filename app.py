from flask import Flask, render_template, request
import cv2
import os
from PIL import Image
import base64
import numpy as np
import shutil
from time import sleep

app = Flask(__name__)

# 定義資料夾路徑
UPLOAD_FOLDER = 'static/original'
PROCESSED_FOLDER = 'static/processed'
CROP_FOLDER = 'static/cropped'
ASSEMBLED_FOLDER = 'static/assembled'
CASCADE_PATH = 'haar_carplate.xml'

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(ASSEMBLED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 上傳的圖片
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)
            
            # 車牌檢測、裁剪和組合
            processed_img_path = process_image(img_path)
            cropped_img_path = crop_plate(img_path)
            assembled_img_path = assemble_characters(cropped_img_path)
            
            # 圖片轉為 base64
            original_img_base64 = convert_to_base64(img_path)
            processed_img_base64 = convert_to_base64(processed_img_path)
            cropped_img_base64 = convert_to_base64(cropped_img_path) if cropped_img_path else None
            assembled_img_base64 = convert_to_base64(assembled_img_path) if assembled_img_path else None
            
            return render_template(
                'index.html',
                original_img=original_img_base64,
                processed_img=processed_img_base64,
                cropped_img=cropped_img_base64,
                assembled_img=assembled_img_base64
            )
    
    return render_template('index.html', original_img=None, processed_img=None, cropped_img=None, assembled_img=None)

def process_image(img_path):
    """車牌檢測並在圖片上畫框"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    signs = detector.detectMultiScale(gray, minSize=(76, 20), scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in signs:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(img_path))
    cv2.imwrite(processed_path, img)
    return processed_path

def crop_plate(img_path):
    """擷取車牌圖像並存儲（灰階 + 二值化）"""
    img = cv2.imread(img_path)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    
    if len(signs) > 0:
        for (x, y, w, h) in signs:
            image = Image.open(img_path)
            cropped = image.crop((x, y, x+w, y+h))
            resized = cropped.resize((140, 40), Image.LANCZOS)
            img_gray = np.array(resized.convert('L'))
            _, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            
            cropped_path = os.path.join(CROP_FOLDER, os.path.basename(img_path))
            cv2.imwrite(cropped_path, img_thre)
            return cropped_path
    else:
        print(f"無法擷取車牌：{os.path.basename(img_path)}")
        return None

def assemble_characters(cropped_img_path):
    """組合車牌字元到黑色背景上"""
    if not cropped_img_path:
        return None
    
    img = cv2.imread(cropped_img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if 5 <= w <= 26 and 20 <= h < 40:  # 過濾噪點
            letter_image_regions.append((x, y, w, h))
    
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    real_shape = []
    for box in letter_image_regions:
        x, y, w, h = box
        letter = thresh[y:y+h, x:x+w]
        real_shape.append(letter)
    
    # 組合字元到黑色背景
    newH = 60
    newW = sum([region.shape[1] for region in real_shape]) + 10 * len(real_shape)
    bg = np.zeros((newH, newW), dtype=np.uint8)
    
    x_offset = 10
    for letter in real_shape:
        h, w = letter.shape
        bg[10:10+h, x_offset:x_offset+w] = letter
        x_offset += w + 10
    
    assembled_path = os.path.join(ASSEMBLED_FOLDER, 'assembled.jpg')
    cv2.imwrite(assembled_path, bg)
    return assembled_path

def convert_to_base64(img_path):
    """將圖片轉為 Base64 編碼"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
