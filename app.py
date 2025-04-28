from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import io
from PIL import Image

# Set the default encoding for stdout to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = './model/mlp_model.h5'
model = load_model(model_path)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Read the image directly from memory
        img = Image.open(file.stream)
        img = img.resize((64, 64))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape
        
        # Predict using the model
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        class_label = 'Stone Detected' if class_index == 0 else 'No Stone'
        
        # Return the prediction result
        return render_template('result.html', label=class_label)

# Route for result page
@app.route('/result')
def result():
    label = request.args.get('label')
    return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

