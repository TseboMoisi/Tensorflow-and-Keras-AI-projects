import numpy as np
from PIL import Image
import scipy.misc 
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

# load model
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = load_model('model.h5')

# load images
# img_names = ['cat-standing.jpg', 'dog.jpg']
# imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
#                    (2, 0, 1)).astype('float32')
#           for img_name in img_names]
# imgs = np.array(imgs) / 255

img_names = ['cat-standing.jpg', 'dog.jpg']
imgs = []

for img_name in img_names:
        # Open image using PIL
        img = Image.open(img_name)
        # Resize image to 32x32
        img = img.resize((32, 32))
        # Convert image to numpy array and normalize to [0. 1]
        img = np.array(img).astype('float32') / 255.0
        # Transpose the image to match the input shape (channels_first)
        img = np.transpose(img, (2, 0, 1)) # Assuming the image has 3 channels (RGB)
        imgs.append(img)
        
imgs = np.array(imgs) # Convert list of images to a numpy array

# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# predict
predictions = model.predict(imgs)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)