from flask import Flask, request, render_template, jsonify

from io import BytesIO
import base64
import numpy as np
from PIL import Image

from tf import softmax_predict, sigmoid_5_layers_predict, relu_5_layers_predict, conv2d_predict

app = Flask('flask-mnist-tensorflow')

app.config.from_pyfile('settings.py')


def decode_img():
    img = request.form['img']
    img = img.split("base64,")[1]
    img = BytesIO(base64.b64decode(img))
    img = Image.open(img) 
    img = Image.composite(img, Image.new('RGB', img.size, 'white'), img)
    img = img.convert('L') 
    img = img.resize((28, 28), Image.ANTIALIAS)  
    img = 1 - np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = decode_img()
    return jsonify({
        'CNN': conv2d_predict(img)[0].tolist(),
        'ReLU': relu_5_layers_predict(img)[0].tolist(),
        'Sigmoid': sigmoid_5_layers_predict(img)[0].tolist(),
        'Softmax': softmax_predict(img)[0].tolist(),
    })


if __name__ == '__main__':
    app.run()
