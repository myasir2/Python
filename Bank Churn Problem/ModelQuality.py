# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

class Evaluator:
    
    def evaluate_model(build_fn, batch_size, epochs, xTrain, yTrain, cv = 10, n_jobs = 1):
        classifier = KerasClassifier(build_fn = build_fn, batch_size = batch_size, epochs = epochs)
        accuracies = cross_val_score(estimator = classifier, X = xTrain, y = yTrain, cv = cv, n_jobs = n_jobs)
        return accuracies
        
class Optimizer:
    
    def optimize_model(build_fn, parameters, xTrain, yTrain, scoring = 'accuracy', cv = 10):
        classifier = KerasClassifier(build_fn = build_fn)
        gridSearch = GridSearchCV(estimator = classifier, 
                          param_grid = parameters,
                          scoring = scoring,
                          cv = cv
                         )
        gridSearch = gridSearch.fit(xTrain, yTrain)
        return gridSearch
