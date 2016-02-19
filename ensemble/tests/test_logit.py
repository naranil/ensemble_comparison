"""
Logitboost Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Here we test the Logitboost classifier we implemented on a randomly generated dataset and for different metrics.
"""

from ensemble.logitboost import BinaryLogitBoost

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

from sklearn.cross_validation import train_test_split

X, y = make_classification(n_features=20, n_samples=200, n_redundant=0, n_informative=10,
                             n_clusters_per_class=1, class_sep=0.9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##### Test Predict and predict proba #####
clf = BinaryLogitBoost(n_estimators=50)
clf.fit(X_train, y_train)
print "==== OUTPUT ===="
print "y_pred", clf.predict(X_test)
print "y_test", y_test

print "==== ACCURACY ===="
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
print "ACC Logit", accuracy_score(y_test, clf.predict(X_test))
print "ACC Ada", accuracy_score(y_test, ada.predict(X_test))

##### Test Cross validation #####
print "Test Cross Validation Vada"
print cross_val_score(clf, X, y, scoring="accuracy")
print "Test Cross Validation Ada"
print cross_val_score(ada, X, y, scoring="accuracy")

##### Test Grid Search #####
parameters = {'n_estimators': np.array([50, 100, 300])}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy")
gs_auc = GridSearchCV(clf, parameters, cv=3, scoring="roc_auc")

gs_acc.fit(X, y)

print "Best parameter p_switch for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores for ACC"
print gs_acc.grid_scores_

gs_auc.fit(X, y)

print "Best parameter p_switch for AUC"
print gs_auc.best_params_
print "Score for AUC with best parameter"
print gs_auc.best_score_
print "All scores for AUC"
print gs_auc.grid_scores_




