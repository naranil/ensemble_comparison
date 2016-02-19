__author__ = 'anilnarassiguin'

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from utils.rms_score import rms_metric_bis

X, y = make_classification(n_features=30, n_samples=1000, n_redundant=0, n_informative=10,
                             n_clusters_per_class=1, class_sep=0.9)

rms = rms_metric_bis()
clf = AdaBoostClassifier(n_estimators=200, algorithm='SAMME.R')
#clf = BaggingClassifier(n_estimators=200)

print cross_val_score(clf, X, y, cv=5, scoring=rms, n_jobs=2)



