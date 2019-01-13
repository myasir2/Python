# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

inputShape = (128, 128)
batchSize = 32

# Init
model = Sequential()

# 1st Layer
model.add(Conv2D(32, (3, 3), input_shape = (*inputShape, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Layer
model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# 3rd Layer
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# 4th Layer
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full Connection
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.6))
model.add(Dense(1, activation = "sigmoid"))

# Compile
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer = adam, loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

trainSet = train_datagen.flow_from_directory('dataset/training_set', 
                                            target_size = inputShape, 
                                            batch_size = batchSize, 
                                            class_mode = 'binary')

testSet = test_datagen.flow_from_directory('dataset/test_set', 
                                            target_size = inputShape, 
                                            batch_size = batchSize, 
                                            class_mode = 'binary')

history = model.fit_generator(trainSet, 
                    steps_per_epoch = 8000/batchSize, 
                    epochs = 42, 
                    validation_data = testSet, 
                    validation_steps = 2000/batchSize,
                    workers = 12)

import numpy as np
from keras.preprocessing import image
testImage = image.load_img('dataset/single_prediction/doc.jpg', target_size = inputShape)
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis = 0)
result = model.predict(testImage)
trainSet.class_indices


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('model.h5')
