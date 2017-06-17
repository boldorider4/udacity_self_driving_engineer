import sys
from sklearn import tree, metrics
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################


#### your code goes here
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)

### be sure to compute the accuracy on the test set
acc = metrics.accuracy_score(labels_test, y_pred)

def submitAccuracies():
  return {"acc":round(acc,3)}
