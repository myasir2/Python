#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:45:58 2018

@author: yasir
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setting independent and dependent vars
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
xDataFrame = pd.DataFrame(x)
y = dataset.iloc[:, 13].values

# Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
labelEncoder = LabelEncoder()
x[:, 1] = labelEncoder.fit_transform(x[:, 1])
x[:, 2] = labelEncoder.fit_transform(x[:, 2])

columnTransformer = make_column_transformer(
                        ([0, 3, 4, 5, 6, 7, 8, 9], StandardScaler()),
                        ([1], OneHotEncoder())
                       )
x = columnTransformer.fit_transform(x)

# Splitting into Train set and Test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Part 2 - ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Add input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compile ann
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fit ann into the test set
classifier.fit(xTrain, yTrain, batch_size = 10, epochs = 100)

# Prediction
yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)

# confusion matrix to measure accuracy
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(yTest, yPred)