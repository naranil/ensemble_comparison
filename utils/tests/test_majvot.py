from __future__ import division

__author__ = 'anilnarassiguin'

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from utils.majvot import predict_majvote_rf

bag = BaggingClassifier(n_estimators=100)
rf = RandomForestClassifier(n_estimators=100)

iris = load_iris()
X = iris['data']
y = iris['target']
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf.fit(X_train, y_train)
#bag.fit(X_train, y_train)

y_soft_rf = rf.predict(X_test)
#y_soft_bag = bag.predict(X_test)

y_hard_rf = predict_majvote_rf(rf, X_test)
#y_hard_bag = predict_majvote_bag(bag, X_test)

print accuracy_score(y_soft_rf, y_test)
assert accuracy_score(y_soft_rf, y_test) == accuracy_score(y_hard_rf, y_test)
#assert accuracy_score(y_soft_rf, y_test) == accuracy_score(y_hard_rf, y_test)