from __future__ import division

__author__ = 'anilnarassiguin'

import numpy as np
from scipy.stats import randint

from utils.rms_score import rms_score

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_features=20, n_samples=200, n_redundant=0, n_informative=10,
                             n_clusters_per_class=1, class_sep=0.9)

rms = make_scorer(rms_score, greater_is_better=True, needs_proba=True)

clf = RandomForestClassifier()

kf = KFold(X.shape[0], n_folds=3, shuffle=True, random_state=1)

gs_rms = GridSearchCV(clf, {'n_estimators' : np.arange(100, 1100, 100)}, cv=kf, scoring=rms, n_jobs=4)
rs_rms = RandomizedSearchCV(clf, {'n_estimators' : randint(100, 1000)}, cv=kf, scoring=rms, n_iter=10, n_jobs=4)

gs_rms.fit(X, y)
rs_rms.fit(X, y)

print "Best parameters -- Grid"
print gs_rms.best_params_
print "Score for ACC with best parameter -- Grid"
print gs_rms.best_score_
print "All scores for ACC -- Grid"
print gs_rms.grid_scores_

print ""
print "Best parameters -- Random"
print rs_rms.best_params_
print "Score for ACC with best parameter -- Random"
print rs_rms.best_score_
print "All scores for ACC -- Random"
print rs_rms.grid_scores_