""" Catedium backend flask API """

import io
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image

app = Flask(__name__)


MODEL_URL = 'https://storage.googleapis.com/catedium-model/catedium/model.h5'
MODEL = None

def load_model_from_url(url):
    """ Load model from google cloud storage """
    global MODEL
    if MODEL is None:
        response = requests.get(url)
        MODEL = load_model(io.BytesIO(response.content))
    return MODEL

def preprocess_image(img):
    """ Preprocess_image before predicting """
    img = img.resize((150, 150))
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """ Doing prediction on the model """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    img = Image.open(file.stream)
    img_array = preprocess_image(img)

    model = load_model_from_url(MODEL_URL)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    breeds = [
        "African Leopard (Panthera pardus)",
        "Caracal (Caracal caracal)",
        "Cheetah (Acinonyx jubatus)",
        "Clouded Leopard (Neofelis nebulosa)", 
        "Jaguar (Panthera onca)",
        "Lion (Panthera leo)",
        "Ocelot (Leopardus pardalis)", 
        "Puma (Puma concolor)",
        "Snow Leopard (Panthera uncia)",
        "Tiger (Panthera tigris)"
    ]

    result = breeds[predicted_class[0]]
    return jsonify({'breed': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
