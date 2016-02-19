"""
Rotation Forest estimator
"""

# Author: Code inspired by Gopi Subramanian on:
# https://www.packtpub.com/books/content/rotation-forest-classifier-ensemble-based-feature-extraction

from __future__ import division

"""
Rotation Forest builds multiple classifiers on randomized projections of the original dataset The feature set is
randomly split into K subsets (K is a parameter of the algorithm) and PCA is applied to each subset in order to create
the training data for the base classifier. The idea of the rotation approach is to encourage simultaneously individual
accuracy and diversity within the ensemble. The size of each subsets of feature was fixed to 3 as proposed by Rodriguez.
The number of sub classes randomly selected for the PCA was fixed to 1 as we focused on binary classification. The size
of the bootstrap sample over the selected class was fixed to 75% of its size.

References
----------
.. [1] Rodriguez J. J., Kuncheva L. I., "Rotation Forest: A New Classifier Ensemble Method" (Pattern Analysis and
       Machine Intelligence, 2006)
"""

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
import numpy as np

from sklearn.externals.six import with_metaclass
from sklearn.ensemble.base import BaseEnsemble
from sklearn.ensemble.forest import BaseForest
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from abc import ABCMeta

class RotationForest(with_metaclass(ABCMeta, BaseEnsemble)):
    """
    Parameters
    ----------
    K : The number of splits in the feature space.

    For the other parameters, please refer to scikit-learn's bagging documentation :
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/bagging.py
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 K=3,
                 verbose=0):
        super(RotationForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.K=K
        self.random_state=random_state
        self.verbose=verbose

    def fit(self, X, y):
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

        self.estimators_  =[]
        self.r_matrices_  =[]

        # Check parameters
        self._validate_estimator()

        models,r_matrices,feature_subsets = self._build_rotationtree_model(X, y)
        self.estimators_ = models
        self.r_matrices_ = r_matrices

        self.classes_ = getattr(models[0], 'classes_', None)
        self.n_classes_ = len(self.classes_)
        #self.feature_subsets_ = feature_subsets

        return self

    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2)) # Binary classification

        estimators = self.estimators_
        r_matrices = self.r_matrices_
        if hasattr(estimators[0], "predict_proba"):
            proba = sum(estimator.predict_proba(X.dot(r)) for estimator, r in zip(estimators, r_matrices))
            normalize = proba.sum(axis=1)
            proba = proba / normalize[:, np.newaxis]
        else:
            for estimator, r in zip(estimators, r_matrices):
                predictions = estimator.predict(X.dot(r)) # Check this part
                for i in range(n_samples):
                    proba[i, predictions[i]] += 1
            normalize = proba.sum(axis=1)
            proba = proba / normalize[:, np.newaxis]
        return proba



    def _get_random_subset(self, iterable):
        k = self.K
        subsets = []
        iteration = 0
        np.random.shuffle(iterable)
        subset = 0
        limit = len(iterable)/k
        while iteration < limit:
            if k <= len(iterable):
                subset = k
            else:
                subset = len(iterable)
            subsets.append(iterable[-subset:])
            del iterable[-subset:]
            iteration+=1
        return subsets

    def _build_rotationtree_model(self, x_train,y_train):
        d = self.n_estimators
        k = self.K
        models = []
        r_matrices = []
        feature_subsets = []
        for i in range(d):
            if self.verbose == 1:
                print("building estimator %d of %d" % (i + 1, d))
            x,_,_,_ = train_test_split(x_train,y_train,test_size=0.3,random_state=7)
            # Features ids
            feature_index = range(x.shape[1])
            # Get subsets of features
            random_k_subset = self._get_random_subset(feature_index)
            feature_subsets.append(random_k_subset)
            # Rotation matrix
            R_matrix = np.zeros((x.shape[1],x.shape[1]),dtype=float)
            for each_subset in random_k_subset:
                pca = PCA()
                x_subset = x[:,each_subset]
                pca.fit(x_subset)
                for ii in range(0,len(pca.components_)):
                    for jj in range(0,len(pca.components_)):
                        R_matrix[each_subset[ii],each_subset[jj]] = pca.components_[ii,jj]

            x_transformed = x_train.dot(R_matrix)

            model = self._make_estimator()
            model.fit(x_transformed,y_train)
            models.append(model)
            r_matrices.append(R_matrix)
        return models,r_matrices,feature_subsets

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(RotationForest, self)._validate_estimator(
            default=DecisionTreeClassifier())


