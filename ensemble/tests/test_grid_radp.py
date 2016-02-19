"""
Random Patches Test
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Random Patches : this method was proposed very recently to tackle the problem of insufficient memory w.r.t. the size of
the data set. The idea is to build each individual model of the ensemble from a random patch of data obtained by drawing
random subsets of both instances and features from the whole dataset.

Here we test Random Patches on a randomly generated dataset and for different metrics with hyperparameter optimization
and different base estimators (Decision Trees and Extremely Randomized Trees).
"""

import numpy as np
from scipy.stats import randint, uniform

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.datasets import make_classification
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import random

radps = {
    'RadP':
    {'clf' : BaggingClassifier(n_jobs=4),
         #'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
         #                'max_samples' : np.arange(0.2, 1.1, 0.1),
         #                'max_features' : np.arange(0.3, 1.1, 0.1) },
         'random_search': {'n_estimators' : np.arange(100, 1100, 100)#,
                           #'max_samples': uniform(0.3, 1),
                           #'max_features': uniform(0.3, 1)
                            }
    },
    'RadPET':
    {    'clf' : BaggingClassifier(base_estimator=ExtraTreeClassifier()),
         #'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
         #                'max_samples' : np.arange(0.2, 1.1, 0.1),
         #                'max_features' : np.arange(0.3, 1.1, 0.1) },
         'random_search': {'n_estimators' : np.arange(100, 1100, 100),
                           'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'max_features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                            }
    }
}

X, y = make_classification(n_features=8, n_samples=100, n_redundant=0, n_informative=6,
                             n_clusters_per_class=1, class_sep=0.9)


param_gs = radps['RadPET']['random_search']
print param_gs

clf = radps['RadPET']['clf']
gs = RandomizedSearchCV(clf, param_gs, cv=3, scoring="accuracy", verbose=1, n_iter=20, n_jobs=2)
gs.fit(X, y)

print "Best parameter for ACC"
print gs.best_params_
print "Score for ACC with best parameter"
print gs.best_score_
print "Grid"
print gs.grid_scores_