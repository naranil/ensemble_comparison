from __future__ import division

__author__ = 'anilnarassiguin'

import numpy as np
from scipy.stats import mode
from sklearn.ensemble.forest import _partition_estimators, _parallel_helper
from sklearn.tree._tree import DTYPE
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

def predict_majvote_rf(forest, X):
    """Predict class for X.

    Uses majority voting, rather than the soft voting scheme
    used by RandomForestClassifier.predict.

    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, n_features]
        The input samples. Internally, it will be converted to
        ``dtype=np.float32`` and if a sparse matrix is provided
        to a sparse ``csr_matrix``.
    Returns
    -------
    y : array of shape = [n_samples] or [n_samples, n_outputs]
        The predicted classes.
    """
    check_is_fitted(forest, 'n_outputs_')

    # Check data
    X = check_array(X, dtype=DTYPE, accept_sparse="csr")

    # Assign chunk of trees to jobs
    n_jobs, n_trees, starts = _partition_estimators(forest.n_estimators,
                                                    forest.n_jobs)

    # Parallel loop
    all_preds = Parallel(n_jobs=n_jobs, verbose=forest.verbose,
                         backend="threading")(
        delayed(_parallel_helper)(e, 'predict', X, check_input=False)
        for e in forest.estimators_)

    # Reduce
    modes, counts = mode(all_preds, axis=0)

    if forest.n_outputs_ == 1:
        return forest.classes_.take(modes[0].astype(int), axis=0)
    else:
        n_samples = all_preds[0].shape[0]
        preds = np.zeros((n_samples, forest.n_outputs_),
                         dtype=forest.classes_.dtype)
        for k in range(forest.n_outputs_):
            preds[:, k] = forest.classes_[k].take(modes[:, k], axis=0)
        return preds
