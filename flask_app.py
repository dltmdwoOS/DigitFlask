from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# CNN 모델 로드
model = load_model('my_model.h5')

# 메인 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')

# 캔버스 이미지 예측 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'})

    image_data = data['image']
    image_data = image_data.split(',')[1]  # 데이터 URL에서 base64 부분만 추출
    image = base64.b64decode(image_data)
    
    img = Image.open(io.BytesIO(image)).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # 모델 입력 형식에 맞게 변환

    # 예측
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
