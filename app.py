import sys
import os
import glob
import re
import uuid
import numpy as np

# Keras
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet_50.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))

    # Preprocessing the image
    x = image.img_to_array(img)

    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    #img_data=preprocess_input(x)
   
    classes=['AadharCard', 'DrivingLicence', 'EmiratesID', 'PanCard', 'VoterID']

    preds_acc = model.predict(x)
    #print(preds_acc)
    preds=np.argmax(preds_acc, axis=1)
    #print(classes[preds[0]])
    acc=str(round(max(preds_acc[0])*100,2))+"%"
    
    return classes[preds[0]]+'('+acc+')'
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("REQUEST:", request.files)
    if request.method == 'POST':
    
        f = request.files.get('file')
        print("F",f.filename)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        if f.filename=="blob":
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(str(uuid.uuid1())+'.jpg'))
        else:
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        print(file_path)
        preds = model_predict(file_path, model)
        result=preds
        print("RESULT : ",result)
        return result
    return None

	



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8005)