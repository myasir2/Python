#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:45:58 2018

@author: yasir
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import TensorBoard

import json

from ModelQuality import Evaluator, Optimizer

def evaluateClassifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    #classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    #classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ['accuracy'])
    return classifier

def optimizeClassifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ['accuracy'])
    return classifier

# Load the data
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
xDataFrame = pd.DataFrame(x)
y = dataset.iloc[:, 13].values

# Configure encoding
labelEncoder = LabelEncoder()
x[:, 1] = labelEncoder.fit_transform(x[:, 1])
x[:, 2] = labelEncoder.fit_transform(x[:, 2])

columnTransformer = make_column_transformer(
                        ([0, 3, 4, 5, 6, 7, 8, 9], StandardScaler()),
                        ([1], OneHotEncoder(categories = "auto"))
                       )
# convert x to float array and encode the data
x = np.vstack(x[:,:]).astype(np.float)
x = columnTransformer.fit_transform(x)

# Split the data into train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Evaluate model
evaluator = Evaluator.evaluate_model(evaluateClassifier, 10, 100, xTrain, yTrain)
accuracyMean = evaluator.mean()
accuracyVariance = evaluator.std()

adam = keras.optimizers.Adam(lr=0.01)
# Optimize model
parameters = {
                "batch_size" : [10, 25, 32],
                "epochs" : [100, 500],
                "optimizer" : [adam]
             }
optimizerGridSearch = Optimizer.optimize_model(optimizeClassifier, parameters, xTrain, yTrain)
bestParameters = optimizerGridSearch.best_params_
bestScore = optimizerGridSearch.best_score_

rmsprop = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=False)
model = optimizeClassifier(adam)
history = model.fit(x = xTrain, y = yTrain, batch_size = 25, epochs = 500, callbacks = [tensorboard])
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model.save('model.h5')
modelJson = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(modelJson, outfile, sort_keys = True, indent = 4)


# Fit ann into the test set
#classifier.fit(xTrain, yTrain, batch_size = 10, epochs = 100)
#
## Prediction
#yPred = classifier.predict(xTest)
#yPred = (yPred > 0.5)
#
#newPredictionData = np.array([[619, 0, 0, 42, 2, 0, 1, 1, 1, 101349]]).astype(np.float)
#newPredictionTransformed = columnTransformer.transform(newPredictionData)
#newPrediction = classifier.predict(newPredictionTransformed)
#newPrediction = (newPrediction > 0.5)
#
## confusion matrix to measure accuracy
#from sklearn.metrics import confusion_matrix
#matrix = confusion_matrix(yTest, yPred)







