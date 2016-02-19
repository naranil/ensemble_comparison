"""
Rotation Forest Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Here we test the Rotation Forest classifier we implemented on a randomly generated dataset and for different metrics.
"""

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd

from utils.rms_score import rms_metric

from ensemble.rotation_forest import RotationForest

def get_data():
    """
    Make a sample classification dataset
    Returns : Independent variable y, dependent variable x
    """
    no_features = 10000
    redundant_features = int(0.1*no_features)
    informative_features = int(0.6*no_features)
    repeated_features = int(0.1*no_features)
    x,y = make_classification(n_samples=500,n_features=no_features,flip_y=0.03,\
            n_informative = informative_features, n_redundant = redundant_features \
            ,n_repeated = repeated_features,random_state=7)
    return x,y

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##### Test Predict and predict proba #####
clf = RotationForest(n_estimators=500, K=3, verbose=1)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)

print "Predict result"
print clf.predict(X_test)

#print "Predict proba result"
#print clf.predict_proba(X_test)
#datasets = pd.HDFStore("../../comparison/data/datasets.h5")
#X = datasets['musk_data']
#y = datasets['musk_target']
rms = rms_metric()
##### Test Cross validation #####
print "Test Cross Validation - ACC"
print np.mean(cross_val_score(clf, X, y, scoring="accuracy"))
print "Test Cross Validation - AUC"
print np.mean(cross_val_score(clf, X, y, scoring="roc_auc"))
print "Test Cross Validation - RMS"
print np.mean(cross_val_score(clf, X, y, scoring=rms))
print "Test Cross Validation - DT - ACC"
print np.mean(cross_val_score(dt, X, y, scoring="accuracy"))
print "Test Cross Validation - RF - ACC"
print np.mean(cross_val_score(rf, X, y, scoring="accuracy"))
print "Test Cross Validation - RF - AUC"
print np.mean(cross_val_score(rf, X, y, scoring="roc_auc"))

# ##### Test Grid Search #####
clf = RotationForest(K=3, verbose=1)
parameters = {'n_estimators': np.arange(100, 1100, 100)}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy", verbose=1)
gs_acc.fit(X, y)

print "Best parameter p_switch for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores"
print gs_acc.grid_scores_

gs_auc = GridSearchCV(clf, parameters, cv=3, scoring="roc_auc", verbose=1)
gs_auc.fit(X, y)

print "Best parameter p_switch for AUC"
print gs_auc.best_params_
print "Score for ACC with best parameter"
print gs_auc.best_score_
print "All scores"
print gs_auc.grid_scores_

rms = rms_metric()
gs_rms = GridSearchCV(clf, parameters, cv=3, scoring=rms, verbose=1)
gs_rms.fit(X, y)

print "Best parameter p_switch for RMS"
print gs_rms.best_params_
print "Score for RMS with best parameter"
print gs_rms.best_score_
print "All scores"
print gs_rms.grid_scores_

##### Test with ET #####
print "CV with ET"
clfET = RotationForest(base_estimator=ExtraTreeClassifier())
print cross_val_score(clf, X, y, scoring="accuracy")