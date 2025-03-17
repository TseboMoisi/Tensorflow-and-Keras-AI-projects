import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2d(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))