import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def main():
    model_path = 'complete.h5'
    image_path = 'c1.jpg'
    model = load_model(model_path)
    test_image = prepare_image(image_path)
    prediction = model.predict(test_image)
    probabilities = tf.nn.softmax(prediction).numpy()
    predicted_class = np.argmax(probabilities)
    class_names = ['class1', 'class2']
    print("Predicted class name:", class_names[predicted_class])

if __name__ == "__main__":
    main()
