# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pylab import bone, pcolor, colorbar, plot, show

from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom

dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, : - 1].values
y = dataset.iloc[:, -1].values

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(x, 100)

# Visualization
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, j in enumerate(x):
    winning_node = som.winner(j)
    plot(winning_node[0] + 0.5,
         winning_node[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Find the frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(7, 8)], mappings[(8, 7)]), axis = 0)
frauds = scaler.inverse_transform(frauds)