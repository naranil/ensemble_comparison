"""
Rotation Boosting (Rotboost) Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Rotboost: this method combines Rot and Adaboost. As the main idea of Rot is to improve the global accuracy of the
classifiers while keeping the diversity through the projections, the idea here is to replace the decision tree by
Adaboost.

Here we test the Rotboost classifier we implemented on a randomly generated dataset and for different metrics.
"""

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np

from utils.rms_score import rms_metric

from ensemble.rotation_forest import RotationForest

def get_data():
    """
    Make a sample classification dataset
    Returns : Independent variable y, dependent variable x
    """
    no_features = 10
    redundant_features = int(0.1*no_features)
    informative_features = int(0.6*no_features)
    repeated_features = int(0.1*no_features)
    x,y = make_classification(n_samples=50,n_features=no_features,flip_y=0.03,\
            n_informative = informative_features, n_redundant = redundant_features \
            ,n_repeated = repeated_features,random_state=7)
    return x,y

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##### Test Predict and predict proba #####
rotb = RotationForest(base_estimator=AdaBoostClassifier(n_estimators=5),n_estimators=40, K=3)
rot = RotationForest(K=3, n_estimators=200)
rf = RandomForestClassifier(n_estimators=200)

#### CV
#print "CV - ACC - RF"
#print np.mean(cross_val_score(rf, X, y, cv=5))

#print "CV - ACC - Rot"
#print np.mean(cross_val_score(rot, X, y, cv=5, scoring="accuracy"))

#print "CV - ACC - RotB"
#print np.mean(cross_val_score(rotb, X, y, cv=5, scoring="accuracy"))

### Tuning rotboost
parameters = {'n_estimators' : np.arange(10, 50, 10),
              'base_estimator' : [AdaBoostClassifier(n_estimators=5), AdaBoostClassifier(n_estimators=10)]}

clf = RotationForest(K=3)
#parameters = {'n_estimators': np.arange(100, 1100, 100)}
gs_acc = GridSearchCV(clf, parameters, cv=3, scoring="accuracy", n_jobs=2)
gs_acc.fit(X, y)

print "Best parameter p_switch for ACC"
print gs_acc.best_params_
print "Score for ACC with best parameter"
print gs_acc.best_score_
print "All scores"
print gs_acc.grid_scores_

rms = rms_metric()
gs_rms = GridSearchCV(clf, parameters, cv=3, scoring=rms, n_jobs=2)
gs_rms.fit(X, y)
print "Best parameter for RMS"
print gs_rms.best_params_
print "Score for RMS with best parameter"
print gs_rms.best_score_
print "All scores"
print gs_rms.grid_scores_











