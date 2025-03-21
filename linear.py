import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense

# Generate a random data
np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20,50, size = len(area))
data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns=['area', 'price'])
plt.scatter(data['area'], data['price'])
plt.show()

# Normalizing the data
data = (data - data.min()) / (data.max() - data.min())

# Creating the model
model = K.Sequential([
    Dense(1, input_shape = [1,], activation=None)
])

model.summary()

# Define our loss function and optimizer algorithm for our model
model.compile(loss='mean_squared_error', optimizer='sgd')

# We are training our model
model.fit(x=data['area'], y=data['price'], epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# Telling our model to predict the house prices
y_pred = model.predict(data['area'])

# Plot a graph with the predicted data and the actual data
plt.plot(data['area'], y_pred, color='red', label="Predicted Price")
plt.scatter(data['area'], data['price'], label="Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()