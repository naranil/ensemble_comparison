"""
Filter used in our experiment:
Signal to Noise ratio (SNR) filter
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

"""
SNR filter.
In the machine learning experiments performed here, features are selected on big databases (more than 1000 features)
----------
"""

from __future__ import division

import numpy as np

def snr(X, y):
    n_features = X.shape[1]
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This function is still not ready for a multiclass classification problem!")

    mean_table = np.zeros((2, n_features))
    sd_table = np.zeros((2, n_features))

    mean_table[0, :] = np.mean(X[y==classes[0]], axis=0)
    mean_table[1, :] = np.mean(X[y==classes[1]], axis=0)

    sd_table[0, :] = np.std(X[y==classes[0]], axis=0)
    sd_table[1, :] = np.std(X[y==classes[1]], axis=0)

    snr_ration = np.abs((mean_table[1, :]-mean_table[0, :]) / (sd_table[1, :] + sd_table[0, :]))

    return np.argsort(snr_ration)[::-1]

