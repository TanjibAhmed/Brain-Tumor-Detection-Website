from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


@app.route('/', methods = ['GET'])
def index():
   return render_template('index.html')

def model_predict(img_path):
   from keras.models import load_model
   from keras.preprocessing import image
   from PIL import Image
   import numpy as np
   import tensorflow as tf
   import cv2

   model = load_model('BrainTumor25Epochs.h5')
   image = cv2.imread(img_path)
   img = Image.fromarray(image)
   img = img.resize((64,64))
   img = np.array(img)
   input_img = np.expand_dims(img, axis=0)
   pred = model.predict(input_img)
   pred = int(model.predict(input_img) ) 
   
   return pred

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      f = request.files['file']
      #saving image to directory
      basepath = os.path.dirname(__file__)
      file_path = os.path.join(
         basepath, 'uploads', secure_filename(f.filename))
      f.save(file_path)
      result = " "
      # Make prediction
      r = model_predict(file_path)
      if r == 1:
         result = "Tumor Cell Detected"
      if r == 0:
         result = "No Tumor Cell Detected"
      return result
   return None 


if __name__ == '__main__':
   app.run(debug = True)