from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('complete.h5')
class_names = ['class1', 'class2']  # Replace with actual class labels

def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            test_image = prepare_image(filepath)
            prediction = model.predict(test_image)
            probabilities = tf.nn.softmax(prediction).numpy()
            predicted_class = np.argmax(probabilities)
            result = class_names[predicted_class]
            return render_template('monitor.html', result=result, image_url=filepath)
    return render_template('monitor.html', result=None, image_url=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
