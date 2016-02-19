"""
ArcX4 estimator
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Arc-x4 is a ensemble-classifier that performs re-weighting of the data points based on the total number of errors that
have occurred for the data point. By its re-weighting procedure, it's a kind of boosting method.
References
----------
.. [1] Breiman L., "Arcing Classifiers" (The Annals of Statistics, 1998)
"""

import numpy as np
from abc import ABCMeta

from sklearn.externals.six import with_metaclass
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.ensemble.forest import BaseForest

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE

from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel, delayed

def _parallel_predict_proba(estimators, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))
    # proba = np.zeros((n_samples, 2)) # to change back to n_classes, this algorithm shoudln't be only binary !
    for estimator in estimators:
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X)

            if len(estimator.classes_) > 1:
                if n_classes == len(estimator.classes_):
                    proba += proba_estimator

                else:
                    proba[:, estimator.classes_] += \
                        proba_estimator[:, range(len(estimator.classes_))]

            # Handle the case where there's only one class in the resulting training set.
            else:
                only_class = estimator.predict(X[0,:])
                if only_class == 1:
                    proba[:, 1] += 1
                else:
                    proba[:, 0] += 1

        else:
            # Resort to voting
            predictions = estimator.predict(X)

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba

class ArcX4(with_metaclass(ABCMeta, BaseEnsemble)):
    """
    Parameters
    ----------
    Please refer to scikit-learn's boosting documentation :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/weight_boosting.py
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 n_jobs=1,
                 verbose=0):
        super(ArcX4, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose

    def fit(self, X, y, sample_weight=None):
        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype)
        n_samples = X.shape[0]

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(n_samples, dtype=np.float)
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
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        weights = np.ones(n_samples)
        weights /= sum(weights)
        missclassed = np.zeros(n_samples)
        estimators = []

        for iboost in range(self.n_estimators):
            # Boosting step
            sample = np.random.choice(n_samples, n_samples, p=weights)
            missclassed, weights, estimator = self._arc_boost(iboost, X, y, missclassed, sample)
            estimators.append(estimator)

        self.estimators_ = estimators

        return self

    def _arc_boost(self, iboost, X, y, missclassed, sample):
        """
        Implement a single boost
        """
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
        X_sample, y_sample = X[sample, :], y[sample]
        estimator.fit(X_sample, y_sample)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            #self.classes_ = np.array([0, 1]) # TO CHANGE
            self.n_classes_ = len(self.classes_)

        y_predict = estimator.predict(X)

        missclassed += y_predict != y

        # Arc-X4 weight formula, cf. article
        weights = (1 + missclassed ** 4)
        weights /= sum(weights)

        return missclassed, weights, estimator

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(ArcX4, self)._validate_estimator(
            default=DecisionTreeClassifier())

