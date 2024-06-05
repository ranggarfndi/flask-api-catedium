from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pyngrok import ngrok
import requests
import os
import time  # Import time module for unique filename generation

# Function to preprocess image
def preprocess_image(image_path, target_size=(150, 150)):
    from PIL import Image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Function to download model from Google Cloud Storage
def download_model(url, dest_path):
    if os.path.exists(dest_path):
        os.remove(dest_path)  # Remove the old model file if it exists
    response = requests.get(url)
    with open(dest_path, 'wb') as f:
        f.write(response.content)

# Define paths
base_dir = 'C:/Users/ACER/flask_api_catedium'
model_path = os.path.join(base_dir, 'model.h5')
train_dir = os.path.join(base_dir, 'train')
uploads_dir = os.path.join(base_dir, 'uploads')
results_dir = os.path.join(base_dir, 'results')

# Ensure the directories exist
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Download the model from Google Cloud Storage
gcs_url = 'https://storage.googleapis.com/catedium-model/catedium/model.h5'
download_model(gcs_url, model_path)

# Load the model
loaded_model_h5 = load_model(model_path)

# Get class labels
class_labels = sorted(os.listdir(train_dir))

app = Flask(__name__)

# Route untuk mengecek apakah API berjalan
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a unique location with a dynamic filename
    unique_filename = f'uploaded_image_{time.time()}.jpg'
    upload_path = os.path.join(uploads_dir, unique_filename)
    file.save(upload_path)

    # Preprocess the image
    img_array = preprocess_image(upload_path)

    # Make prediction
    prediction = loaded_model_h5.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = class_labels[predicted_class]

    # Save the result image in the results directory
    result_filename = f'result_{unique_filename}'
    result_path = os.path.join(results_dir, result_filename)
    os.rename(upload_path, result_path)

    # Return JSON response
    return jsonify({
        'class': class_name,
        'image_path': result_path  # Returning the path to the saved image
    })

# Set ngrok authtoken and start ngrok
authtoken = "2h3Oz432OGSdTgS1P92qoJ7oc0C_CXpAc1EPr4pqsfmR8Mce"  # Replace with your actual authtoken
ngrok.set_auth_token(authtoken)

# Open an HTTP tunnel on the default port 5000
public_url = ngrok.connect(5000, bind_tls=True)
print(f"Public URL: {public_url}")

# Run the app
if __name__ == '__main__':
    app.run()
