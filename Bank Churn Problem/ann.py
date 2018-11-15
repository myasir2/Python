#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:45:58 2018

@author: yasir
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
xDataFrame = pd.DataFrame(x)
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
labelEncoder = LabelEncoder()
x[:, 1] = labelEncoder.fit_transform(x[:, 1])
x[:, 2] = labelEncoder.fit_transform(x[:, 2])

columnTransformer = make_column_transformer(
                        ([0, 3, 4, 5, 6, 7, 8, 9], StandardScaler()),
                        ([1], OneHotEncoder())
                       )

#columnTransformer = make_column_transformer(
#        ([1], OneHotEncoder()))

x = columnTransformer.fit_transform(x)
