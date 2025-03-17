import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Generate a random data
np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20,50, size = len(area))

# Create a pandas DataFrame
data = pd.DataFrame({'area': area, 'price': price})


# Calculate the two regression coefficients using the equations we defined
W = sum(price*(area - np.mean(area))) / sum((area - np.mean(area))**2)
b = np.mean(price) - W * np.mean(area)
print("The regression coefficients are", W, b)

# Prediction for new prices
y_pred = W * area + b

# Here we plot the predicted prices with the actual price
plt.plot(area, y_pred, color='red', label="Predicted Price")
plt.scatter(data['area'], data['price'], label="Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()