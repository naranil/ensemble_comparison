"""
Variance Penalizing Adaboost (Vadaboost) estimator
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
VadaBoost is similar to AdaBoost except that the weighting function tries to minimize both empirical risk and empirical variance.
This modification is motivated by the recent empirical bound which relates the empirical variance to the true risk.

References
----------
.. [1] Shivaswamy P.K., Jebara T., "Variance Penalizing Adaboost" (Advances in Neural Information Processing Systems 24
       , 2011)
"""

import numpy as np
from abc import ABCMeta

from sklearn.externals.six import with_metaclass

from sklearn.ensemble.base import BaseEnsemble

from sklearn.tree._tree import DTYPE
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import column_or_1d, check_X_y
from sklearn.utils.multiclass import check_classification_targets

class BinaryVadaboost(with_metaclass(ABCMeta, BaseEnsemble)):
    """
    Parameters
    ----------
    lbda : scalar parameter for regularization

    For the other parameters, please refer to scikit-learn's boosting documentation :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/weight_boosting.py
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 lbda=0.1):
        super(BinaryVadaboost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.lbda=lbda
        self.random_state=random_state

    def fit(self, X, y, sample_weight=None):
        y = self._validate_y(y)

        X, y = check_X_y(X, y, accept_sparse='csc', dtype=DTYPE)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
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
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        #self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        estimators = []
        estimator_weights = []

        for iboost in range(self.n_estimators):
            estimator, sample_weight, alpha, breaked = self._boost(X, y, sample_weight)
            if breaked and alpha == 0:
                break
            else:
                estimators.append(estimator)
                estimator_weights.append(alpha)
                if breaked:
                    break
                #estimator_errors.append(alpha)

        self.estimators_ = estimators
        self.estimator_weights_ = estimator_weights

        return self



    def _boost(self, X, y, sample_weight):
        breaked = False
        estimator = self._make_estimator()
        n_samples = X.shape[0]

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        reg_weights = self.lbda*n_samples*(sample_weight**2) + (1 - self.lbda)*sample_weight

        estimator.fit(X, y, sample_weight=reg_weights)

        y_predict = estimator.predict(X)
        trues = reg_weights[y == y_predict]
        falses = reg_weights[~(y == y_predict)]

        if len(falses) == 0:
            alpha = (1/4) * np.log(sum(trues) / 0.01)
            breaked = True
        else:
            alpha = (1/4) * np.log(sum(trues) / sum(falses))


        y_predict = estimator.predict(X)
        y_transform = 2*y-1
        y_pred_transform = 2*y_predict-1
        sample_weight *= np.exp(-y_transform*y_pred_transform*alpha)
        sample_weight /= sum(sample_weight)

        if alpha > 0:
            return estimator, sample_weight, alpha, breaked
        else:
            breaked = True
            return estimator, sample_weight, alpha, breaked

    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2)) # Binary classification

        proba = sum(estimator.predict_proba(X) * w
                for estimator, w in zip(self.estimators_,
                                        self.estimator_weights_))
        normalize = proba.sum(axis=1)
        proba = proba / normalize[:, np.newaxis]

        return proba

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        if n_classes > 2:
            raise ValueError("It's a binary classification algorithm. Use a dataset with only 2 classes to predict.")

        return y

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BinaryVadaboost, self)._validate_estimator(
            default=DecisionTreeClassifier())
