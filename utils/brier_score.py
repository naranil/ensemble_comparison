from __future__ import division

__author__ = 'anilnarassiguin'

from sklearn.metrics import brier_score_loss

import numpy as np

def rms_score(y_true, probas, pos_label=1):
    if pos_label <= 0:
        y_prob = probas[:, 0]
    else:
        y_prob = probas[:, 1]

    return 1 - np.average((y_true-y_prob)**2)
