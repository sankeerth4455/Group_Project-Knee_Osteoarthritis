# app.py
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model(r"vgg19_model.h5")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the form
        file = request.files['image']
        # Preprocess the image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224,224))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Encode image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_src = "data:image/jpeg;base64," + img_str
        return render_template('home.html', prediction=predicted_class, img_src=img_src)

if __name__ == '__main__':
    app.run(debug=True)
