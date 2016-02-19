"""
Calibration Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Here we test the scikit's learn calibration module.
"""

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

import numpy as np
import pandas as pd

from utils.rms_score import rms_metric_bis

def get_data():
    """
    Make a sample classification dataset
    Returns : Independent variable y, dependent variable x
    """
    no_features = 30
    redundant_features = int(0.1*no_features)
    informative_features = int(0.6*no_features)
    repeated_features = int(0.1*no_features)
    x,y = make_classification(n_samples=500,n_features=no_features,flip_y=0.03,\
            n_informative = informative_features, n_redundant = redundant_features \
            ,n_repeated = repeated_features,random_state=7)
    return x,y

X, y = get_data()
#data = pd.HDFStore("../../comparison/data/datasets.h5")
#X = data['colon_data']
#y = data['colon_target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
rms = rms_metric_bis()

#clf = AdaBoostClassifier(n_estimators=50, random_state=1)

# print "----- WITHOUT CALIBRATION -----"
# print cross_val_score(clf, X_train, y_train, cv=5)
# print ""

print "----- WITH CALIBRATION -----"
clf = BaggingClassifier(n_estimators=600, random_state=1)
cal = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv=2)
cal.fit(X_val, y_val)
print "ACC"
print cross_val_score(cal, X_train, y_train, cv=5)
print "AUC"
print cross_val_score(cal, X_train, y_train, cv=5, scoring="roc_auc")
print "RMS"
print cross_val_score(cal, X_train, y_train, cv=5, scoring=rms)