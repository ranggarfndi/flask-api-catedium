from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pyngrok import ngrok
import os
# import time  # Tambahkan ini untuk mengimpor modul time

# Function to preprocess image
def preprocess_image(image_path, target_size=(150, 150)):
    from PIL import Image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Load the model
model_path = 'C:/Users/ACER/flask_api_catedium/model.h5'
loaded_model_h5 = load_model(model_path)

# Get class labels
base_dir = 'C:/Users/ACER/flask_api_catedium'
train_dir = os.path.join(base_dir, 'train')
class_labels = sorted(os.listdir(train_dir))

app = Flask(__name__)

# Route untuk mengecek apakah API berjalan
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    # Define upload_path within the function
    upload_path = 'C:/Users/ACER/flask_api_catedium/uploads/uploaded_image.jpg'

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a unique location with a dynamic filename
    upload_dir = 'C:/Users/ACER/flask_api_catedium/uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    # Generate a unique filename based on current time
    # upload_path = os.path.join(upload_dir, f'uploaded_image_{time.time()}.jpg')
    file.save(upload_path)  # Save the file using the defined upload_path

    # Preprocess the image
    img_array = preprocess_image(upload_path)

    # Make prediction
    prediction = loaded_model_h5.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = class_labels[predicted_class]

    # Return JSON response
    return jsonify({
        'class': class_name
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
