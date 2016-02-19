"""
Arc-X4 Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

"""
Here we test the Arc-X4 classifier we implemented on the iris dataset and for different metrics.
"""

from ensemble.arc_x4 import ArcX4

import numpy as np

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.tree import ExtraTreeClassifier


iris = load_iris()
X = iris['data']
y = iris['target']
X, y = shuffle(X, y)

##### Test Predict and predict proba #####
clf = ArcX4()
clf.fit(X, y)
print "Predict result"
print clf.predict(X)

print "Predict proba result"
print clf.predict_proba(X)

##### Test Cross validation #####
print "Test Cross Validation"
print cross_val_score(clf, X, y, scoring="accuracy")

##### Test Grid Search #####
parameters = {'n_estimators': np.array([50, 200, 1000])}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy")
#gs_auc = GridSearchCV(clf, parameters, cv=3, scoring="roc_auc")

gs_acc.fit(X, y)

print "Best parameter p_switch for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores"
print gs_acc.grid_scores_

##### Test Grid Search #####
print "CV with ET"
clfET = ArcX4(base_estimator=ExtraTreeClassifier())
print cross_val_score(clf, X, y, scoring="accuracy")