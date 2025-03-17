import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np

# Prebuild model with pre-trained weights on imagenet
base_model = VGG16(weights='imagenet', include_top=True)
print(base_model.summary())

# Safer way to print layer information
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

# Extract features from block4_pool
model = models.Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# Check if image file exists before proceeding
import os
img_path = 'cat.jpg'
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found.")
    # You could optionally download a sample image here
    # from urllib.request import urlretrieve
    # urlretrieve("https://sample-url.com/cat.jpg", "cat.jpg")
else:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the features from this block
    features = model.predict(x)
    print(f"Feature shape: {features.shape}")
    print(features)