import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Normalization
import seaborn as sns

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']

data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
data = data.drop('origin', axis=1)
print(data.isna().sum())
data = data.dropna()

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

# Visualize the relationship between the different variables
 # sns.pairplot(train_dataset[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']], diag_kind='kde')

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('mpg')
test_labels = test_features.pop('mpg')

# Normalize
data_normalizer = Normalization(axis=1)
data_normalizer.adapt(np.array(train_features))

model = K.Sequential([
    data_normalizer,
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation=None)
])
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x=train_features,y=train_labels, epochs=100, verbose=1, validation_split=0.2)

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Error [MPG]')
# plt.legend()
# plt.grid(True)

# Get predictions
y_pred = model.predict(test_features).flatten()

# Create a plot
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_labels, c='blue', label='True Vales', alpha=0.5)
plt.scatter(test_labels, y_pred, c='red', label='Predictions', alpha=0.5)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, 'k--', label='Ideal')
plt.legend()
plt.show()