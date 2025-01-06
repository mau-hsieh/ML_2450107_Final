import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# 定義 MLP 模型類
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.softmax(output_input)
        return output

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

    def load_model(self, file_path):
        data = np.load(file_path)
        self.weights_input_hidden = data["weights_input_hidden"]
        self.bias_hidden = data["bias_hidden"]
        self.weights_hidden_output = data["weights_hidden_output"]
        self.bias_output = data["bias_output"]

# 初始化 MLP 模型
mlp_model = MLP(input_size=54, hidden_size=64, output_size=10)

# Streamlit App 界面設置
st.title("手寫數字識別")

# 讓使用者選擇是否加載模型
model_choice = st.selectbox("選擇模型加載方式", ("自動加載預設模型 Cross_Entropy :","自動加載預設模型 MSE :", "手動上傳模型"))

if model_choice == "自動加載預設模型 Cross_Entropy :":
    # 嘗試自動加載預設的 MLP 模型
    model_path = "mlp_model09.npz"  # 預設模型文件
    if os.path.exists(model_path):
        mlp_model.load_model(model_path)
        st.success(f"已自動加載預設模型：{model_path}")
    else:
        st.warning("未找到預設模型文件，請上傳模型")

elif model_choice == "自動加載預設模型 MSE :":
        # 嘗試自動加載預設的 MLP 模型
    model_path = "mlp_model_mse_09.npz"  # 預設模型文件
    if os.path.exists(model_path):
        mlp_model.load_model(model_path)
        st.success(f"已自動加載預設模型：{model_path}")
    else:
        st.warning("未找到預設模型文件，請上傳模型")

elif model_choice == "手動上傳模型":
    model_file = st.file_uploader("上傳 MLP 模型 (.npz)", type=["npz"])
    if model_file:
        mlp_model.load_model(model_file)
        st.success("模型已成功加載")

# 選擇上傳圖片或即時手寫
option = st.selectbox("選擇輸入方式", ("即時手寫", "上傳圖片"))

if option == "上傳圖片":
    uploaded_file = st.file_uploader("請上傳手寫數字圖片", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="上傳的原始圖片", use_column_width=True)

        # 圖片處理為 9x6 的黑白圖像
        image = ImageOps.autocontrast(image)
        binary_image = image.point(lambda p: 255 if p > 180 else 0)
        binary_cv_image = np.array(binary_image)
        
        x, y, w, h = cv2.boundingRect(binary_cv_image)
        cropped_image = binary_cv_image[y:y+h, x:x+w]
        
        if cropped_image.size != 0:
            resized_image = cv2.resize(cropped_image, (6, 9), interpolation=cv2.INTER_NEAREST)
            st.image(resized_image, caption="處理後的 9x6 黑白圖像", width=120)
            
            flattened_image = resized_image.flatten().reshape(1, -1)
            if st.button("預測"):
                predicted_digit = mlp_model.predict(flattened_image)[0]
                st.write(f"預測結果：{predicted_digit}")
        else:
            st.warning("無法找到有效的前景數字，請上傳包含數字的圖片")

elif option == "即時手寫":
    canvas = st_canvas(
        background_color="black",
        stroke_color="white",
        stroke_width=20,
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img_array = (canvas.image_data[:, :, :3] * 255).astype(np.uint8)
        img_array[img_array.sum(axis=2) == 0] = [0, 0, 0]  # 設置為黑色
        img_array[img_array.sum(axis=2) != 0] = [255, 255, 255]  # 設置為白色
        image = Image.fromarray(img_array, "RGB").convert("L")

        binary_image = image.point(lambda p: 255 if p > 127 else 0)
        binary_cv_image = np.array(binary_image)

        x, y, w, h = cv2.boundingRect(binary_cv_image)
        cropped_image = binary_cv_image[y:y+h, x:x+w]

        if cropped_image.size != 0:
            resized_image = cv2.resize(cropped_image, (6, 9), interpolation=cv2.INTER_NEAREST)
            flattened_image = resized_image.flatten().reshape(1, -1)
            if st.button("預測"):
                predicted_digit = mlp_model.predict(flattened_image)[0]
                st.write(f"預測結果：{predicted_digit}")
            
            st.image(resized_image, caption="處理後的 9x6 黑白圖像", width=120)
            
            

        else:
            st.warning("請在畫布上繪製數字再進行處理")
