from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # 將文件讀取為二進制數據
            img_bytes = file.read()
            # 將二進制數據轉換為 base64 編碼
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            # 在頁面上顯示圖片
            return render_template('index.html', img_data=img_base64)
    return render_template('index.html', img_data=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
