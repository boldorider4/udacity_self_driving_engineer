import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree, metrics

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively


def submitAccuracies():
    clf_min2 = tree.DecisionTreeClassifier(min_samples_split=2)
    clf_min50 = tree.DecisionTreeClassifier(min_samples_split=50)
    clf_min2.fit(features_train, labels_train)
    clf_min50.fit(features_train, labels_train)

    pred2 = clf_min2.predict(features_test)
    pred50 = clf_min50.predict(features_test)
    
    acc_min_samples_split_2 = metrics.accuracy_score(labels_test, pred2)
    acc_min_samples_split_50 = metrics.accuracy_score(labels_test, pred50)

    return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3), \
            "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
