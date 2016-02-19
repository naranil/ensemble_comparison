"""
Logitboost estimator
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
LogitBoost is a boosting classification algorithm. LogitBoost and AdaBoost are close to each other in the sense that
both perform an additive logistic regression. The difference is that AdaBoost minimizes the exponential loss, whereas
LogitBoost minimizes the logistic loss.
For the need of our experiments, we only implemented a binary version of LogitBoost.
References
----------
.. [1] Friedman J., Hastie T., Tibshirani R., "Additive logistic regression: a statistical view of boosting" (Annals of
       Statistics, 2000)
"""

import numpy as np
from abc import ABCMeta

from sklearn.externals.six import with_metaclass

from sklearn.ensemble.base import BaseEnsemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE

from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils.multiclass import check_classification_targets

class BinaryLogitBoost(with_metaclass(ABCMeta, BaseEnsemble)):
    """
    Parameters
    ----------
    Please refer to scikit-learn's boosting documentation :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/weight_boosting.py
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None):
        super(BinaryLogitBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.random_state=random_state

    def fit(self, X, y, sample_weight=None):
        y = self._validate_y(y)

        X, y = check_X_y(X, y, accept_sparse='csc', dtype=DTYPE)
        n_samples = X.shape[0]

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(n_samples, dtype=np.float)
            sample_weight[:] = 1. / n_samples
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []

        estimators = []
        predictions = np.zeros(n_samples)

        p = 0.5*np.ones(n_samples)

        for iboost in range(self.n_estimators):
            sample_weight = p * (1 - p)
            z = (y-p) / sample_weight

            estimator = self._make_estimator()

            try:
                estimator.set_params(random_state=self.random_state)
            except ValueError:
                pass

            estimator.fit(X, z, sample_weight=sample_weight)
            estimators.append(estimator)
            predictions += (1/2)*estimator.predict(X)

            p = 1 / (1 + np.exp(-2*predictions))

        self.estimators_ = estimators

        return self

    def predict(self, X):
        estimators = self.estimators_
        predictions = sum([estimator.predict(X) for estimator in estimators])
        return self.classes_.take(np.where(predictions > 0, 1, 0))

    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2)) # Binary classification

        estimators = self.estimators_

        predictions = sum([estimator.predict(X) for estimator in estimators])
        proba[:, 0] = 1 / (1+np.exp(predictions))
        proba[:, 1] = 1-proba[:, 0]

        return proba



    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BinaryLogitBoost, self)._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        if n_classes > 2:
            raise ValueError("It's a binary classification algorithm. Use a dataset with only 2 classes to predict.")

        return y