import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from imagerg import N_HIDDEN, VALIDATION_SPLIT

# Network and training
EPOCHS = 64 # The amount of times the model is exposed to the training set
BATCH_SIZE = 128 # The number of training instance that are going to happen before optimizer performs a weight update
VERBOSE = 1 # The amount of information show using the training process in the terminal
NB_CLASSES = 10 # number of outputs/number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much of the training dataset is reserved of for VALIDIATION
DROPOUT = 0.3
# Loading MNIST dataset
# Labels have one-hot representation
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() # Here we are loading the data set and splitting into training sets and test sets

# X_train set has 60,000 rows of (28x28) pixel pictures or values; we reshape it to 60,000 x 784
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize inputs to be within in [0,1]
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Labels have one-hot representation.
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
            input_shape=(RESHAPED,),
            name='dense_layer', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
            name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
            name='dense_layer_3', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
            name='dense_layer_4', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(64, input_dim=64, kernel_regularizer=l2(0.01),
                activity_regularizer=l2(0.01)))
model.add(keras.layers.Dense(NB_CLASSES,
            name='dense_layer_5', activation='softmax'))

# Summary of the model
model.summary()

# Compiling the model
model.compile(optimizer='RMSProp',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model.
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluating the model.
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)