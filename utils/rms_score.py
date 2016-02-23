"""
1-RMS score function
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Measure the accuracy of probabilistic predictions. (cf . Brier score)
----------
"""

__author__ = 'anilnarassiguin'

from sklearn.metrics import make_scorer

import numpy as np

def rms_score(y_true, probas, pos_label=1):
    if pos_label <= 0:
        y_prob = probas[:, 0]
    else:
        y_prob = probas[:, 1]

    return 1 - np.sqrt(np.average((y_true-y_prob)**2))

def rms_metric():
    rms = make_scorer(rms_score, greater_is_better=True, needs_proba=True)
    return rms
