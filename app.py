# app.py

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('Skin Cancer Classification Model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file)
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        result = model.predict(np.expand_dims(image, axis=0))
        class_index = np.argmax(result)
        return jsonify({'class': class_index})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main':
    app.run(host='0.0.0.0', port=5000)
