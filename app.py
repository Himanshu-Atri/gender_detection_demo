from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model = load_model("gender_detector.keras")

# Create Flask app instance
app = Flask(__name__)

# Set a folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to process the uploaded image
def get_result(img_path):
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (224, 224))
    img_resize = np.array(img_resize, dtype=np.float32)
    img_resize /= 255.0
    img_input = img_resize.reshape(1, 224, 224, 3)
    prediction = model.predict(img_input)

    if prediction[0][0] < 0.5:
        return "He is a Men."
    else:
        return "She is a Women."

# Home route, returns an HTML form for uploading images
@app.route('/')
def home():
    return render_template('index.html')

# Process route to handle the image upload
@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Get the result for the uploaded image
        result = get_result(file_path)
        
        return f"Prediction: {result}"

if __name__ == '__main__':
    app.run(debug=True)
