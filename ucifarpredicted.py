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
img_names = ['cat-standing.jpg', 'dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                    (2, 0, 1)).astype('float32')
            for img_name in img_names]
imgs = np.array(imgs) / 255

# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# predict
predictions = model.predict_classes(imgs)
print(predictions)