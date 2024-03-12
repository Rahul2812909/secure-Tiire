from flask import Flask, request, render_template
import numpy as np
import pickle
from keras_preprocessing import image
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageTk
import os
from keras_preprocessing.image import ImageDataGenerator



app = Flask(__name__)

model = tf.keras.models.load_model('MODEL_.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.files['image']
    #print(data)
    image_path = os.path.join('static/uploads', data.filename)
    data.save(image_path)
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    # Get the predicted class label (index of maximum probability)
    predicted_class_index = np.argmax(result)
    image_filename = data.filename

    # Define the class labels
    class_labels = ['Tiretread is 1.5mm', 'Tiretread is 4mm', 'Tiretread is 6mm', 'Tiretread is 8mm']
    classs = class_labels[predicted_class_index]    
    
    prob = result[0, predicted_class_index]
    
    return render_template('predict.html', classs=classs, prob=prob, image_filename=image_filename)


if __name__ == '__main__':
    app.run(debug=False, port=800)



    

