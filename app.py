from flask import Flask , render_template , request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from utils import process_image
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

model = tf.keras.models.load_model(r"D:\AI\projects\deep fake detection\fine_tuned_model.keras")

Threshold = 0.7

@app.route("/" , methods = ['GET' , 'POST'])

def home():
    prediction = None 
    probability = None

    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file).convert('RGB')
        processed = process_image(image)

        probability = model.predict(processed , verbose = 0)[0][0]

        if probability >= 0.7:
            prediction  = 'REAL IMAGE'
        else:
            prediction = 'AI GENERATED IMAGE'

    return render_template(
        "index.html",
        prediction = prediction  
    )

if __name__ == "__main__":
    app.run(debug = True)


