from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import time
MODEL_PATH = 'model/flowerclassification.h5'
model = load_model(MODEL_PATH)

# Define a flask app
app = Flask(__name__)
classes=["Hoa cúc susan mắt đen","Hoa chuông xanh","Hoa mao lương","Hoa cúc","Hoa khô Colstfoot","Hoa thanh cúc","Hoa anh thảo","Hoa nghệ tây","Hoa thủy tiên","Hoa cúc dại","Hoa bồ công anh","Hoa dâm bụt","Hoa diên vĩ","Hoa oải hương","Hoa linh lan","Hoa sen","Hoa lan","Hoa Pansy","Hoa giọt tuyết","Hoa hướng dương","Hoa loa kèn","Hoa Tulips","Windflower","Hoa hồng"]

#Thực hiện predict
def model_predict(img_path, model):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (180, 180))
    x = np.array(img).astype("float16")
    x = x/255
    x = x.reshape(-1, 180, 180, 3)
    tuan = model.predict(x)
    return tuan


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
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Thực hiện predict
        tuan = model_predict(file_path, model)
        final = (int)(np.argmax(tuan))

        #Nếu xác suất lớn hơn 0.8 thì sẽ là 1 trong 24 loài hoa
        if max(tuan[0])>0.8:
            result = classes[final]
        else:
            result = "Không thể phân biệt"
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

