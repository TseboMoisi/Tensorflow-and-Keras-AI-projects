import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input  # Import the Input layer

# Constants
NB_CLASSES = 10  # Number of output neurons
RESHAPED = 784   # Number of input features

# Define the model
model = tf.keras.models.Sequential()

# Add an Input layer to specify the input shape
model.add(Input(shape=(RESHAPED,), name='input_layer'))

# Add the Dense layer
model.add(keras.layers.Dense(NB_CLASSES,
                             kernel_initializer='zeros',
                             name='dense_layer',
                             activation='softmax'))

# Compile the model (optional, but recommended for training)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()