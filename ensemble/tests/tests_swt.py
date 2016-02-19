"""
Class Switching Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

"""
Here we test the Class Switching classifier we implemented on a randomly generated dataset and for different metrics.
"""

import sys
sys.path.append("../..")

from ensemble.class_switching import ClassSwitching

import numpy as np

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.tree import ExtraTreeClassifier


iris = load_iris()
X = iris['data']
y = iris['target']
X, y = shuffle(X, y)

p = 0.1

##### Test Predict and predict proba #####
clf = ClassSwitching(p_switch=p)
clf.fit(X, y)
print "Predict result"
print clf.predict(X)

print "Predict proba result"
print clf.predict_proba(X)

##### Test Cross validation #####
print "Test Cross Validation - ACC"
print cross_val_score(clf, X, y, scoring="accuracy")
print "Test Cross Validation - 1-RMS"
print cross_val_score(clf, X, y, scoring="brier_score_loss")

##### Test Grid Search #####
parameters = {'p_switch': np.array([0.01, 0.05, 0.1, 0.2])}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy")
gs_auc = GridSearchCV(clf, parameters, cv=3, scoring="roc_auc")

gs_acc.fit(X, y)

print "Best parameter p_switch for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores"
print gs_acc.grid_scores_

##### Test Grid Search #####
print "CV with ET"
clfET = ClassSwitching(base_estimator=ExtraTreeClassifier())
print cross_val_score(clf, X, y, scoring="accuracy")

#### TEST Swt on small datasets ####

# data = pd.HDFStore("../../comparison/data/datasets.h5")
#
# print "==== FOR OVARIAN ===="
# X = data['leukemia_data']
# y = data['leukemia_target']
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
# clf = ClassSwitching()
#
# parameters = {'p_switch': np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4])}
# gs_acc = GridSearchCV(clf, parameters, cv=2, scoring="accuracy")
#
# gs_acc.fit(X_val, y_val)
# print "Best parameter p_switch for ACC"
# print gs_acc.best_params_
# print "Score for ACC with best parameter"
# print gs_acc.best_score_
# print "All scores"
# print gs_acc.grid_scores_
#
# print "==== FOR LEUKEMIA ===="
# X = data['leukemia_data']
# y = data['leukemia_target']
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
# print X_val.shape
# clf = ClassSwitching()
#
# parameters = {'p_switch': np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4])}
# gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy")
#
# gs_acc.fit(X_val, y_val)
# print "Best parameter p_switch for ACC"
# print gs_acc.best_params_
# print "Score for ACC with best parameter"
# print gs_acc.best_score_
# print "All scores"
# print gs_acc.grid_scores_







