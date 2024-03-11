import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

# Load the VGG16 model pre-trained on ImageNet data
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess the image
img_path = 'img/MK.jpg'  # Change this to the path of your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get the embeddings from one of the intermediate layers
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('block5_conv2').output)
embeddings = intermediate_layer_model.predict(img_array)

# Now 'embeddings' contains the embeddings for the image
print(embeddings.shape)
