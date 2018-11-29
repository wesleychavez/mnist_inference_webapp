from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.hdf5'

# Load your trained model
global model
graph = tf.get_default_graph()
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, color_mode = "grayscale", target_size=(28, 28))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    print (x.shape)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # Convert to string
        result = "I'm " + str(np.max(preds)*100) + "% certain it's a " + str(np.argmax(preds)) + "."               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
