"""
Variance penalizing Adaboost (Vadaboost) Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

"""
Here we test the Vadaboost classifier we implemented on a randomly generated dataset and for different metrics.
"""

import sys
sys.path.append("..")

from ensemble.vadaboost import BinaryVadaboost

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss, make_scorer
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

from utils.brier_score import rms_score
from utils.rms_score import rms_metric

X, y = make_classification(n_features=20, n_samples=200, n_redundant=0, n_informative=10,
                             n_clusters_per_class=1, class_sep=0.9)

##### Test Predict and predict proba #####
clf = BinaryVadaboost(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=1), lbda=0.1)
ada = AdaBoostClassifier()
clf.fit(X, y)
# ada.fit(X, y)
print ada.estimators_
print "Predict result"
print clf.predict(X)
print "Accuracy"
print accuracy_score(y, clf.predict(X))

print "Predict proba result"
y_prob = clf.predict_proba(X)
#print y_prob
print "rms score"
rms = rms_metric()
#print brier_score_loss(y, y_prob[:, 1])


##### Test Cross validation #####
print "Test Cross Validation Vada"
print cross_val_score(clf, X, y, scoring="accuracy")
print "Test Cross Validation Ada"
print cross_val_score(ada, X, y, scoring="accuracy")

rms = rms_metric()
##### Test CV Brier #####
print "Test Cross Validation Vada - Brier"
print cross_val_score(clf, X, y, scoring=rms)
print "Test Cross Validation Ada - Bier"
print cross_val_score(ada, X, y, scoring=rms)

##### Test Grid Search #####
parameters = {'n_estimators': np.array([50, 100, 300]), 'lbda': np.array([0.1, 0.3, 0.5, 0.7, 0.9])}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy")
gs_auc = GridSearchCV(clf, parameters, cv=3, scoring="roc_auc")
gs_rms = GridSearchCV(clf, parameters, cv=3, scoring=rms)

gs_acc.fit(X, y)

print "Best parameter lbda for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores for ACC"
print gs_acc.grid_scores_

gs_auc.fit(X, y)

print "Best parameter lbda for AUC"
print gs_acc.best_params_
print "Score for AUC with best parameter"
print gs_acc.best_score_
print "All scores for AUC"
print gs_acc.grid_scores_

gs_rms.fit(X, y)