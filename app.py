from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import io
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)

# Move model loading inside predict route
model = None
model_path = './model/mlp_model.h5'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model(model_path)  # Load model when first prediction request comes

    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        img = Image.open(file.stream)
        img = img.resize((64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        class_label = 'Stone Detected' if class_index == 0 else 'No Stone'
        
        return render_template('result.html', label=class_label)

@app.route('/result')
def result():
    label = request.args.get('label')
    return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
