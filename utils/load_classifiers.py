"""
Functions to load classifiers calling their nicknames (ex: load_classfiers('RF') to call Random Forest)
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Use those functions to call all the following classifiers:
Bag, BagET : Bagging classifier and its ET (with extra-tree classifier as base learner) version - from sklearn
RF: Random Forest classifier - from sklearn
RadP, RadPET: Random Patches classifier - using Bagging from sklearn
Ad, AdSt, AdET: Adaboost and its stump and ET version - from sklearn
Vad, VadET: Variance penalizing Adaboost - our implementation
Swt, SwtET: Class switching classifier - our implementation
ArcX4, ArcX4ET: Arcing classifier - our implementation
Logb: Logitboost - our implementation
Rot, RotET: Rotation Forest - our implementation
Rotb, RotbET: Rotation Boosting - our implementation
CART: Decision Tree classifier - from sklearn
ET: Extra tree classifier - from sklearn

The grid search paramaters to tune the classifiers are also given in this code.
----------
"""

import numpy as np
from scipy.stats import randint, uniform

from ensemble.vadaboost import BinaryVadaboost
from ensemble.class_switching import ClassSwitching
from ensemble.arc_x4 import ArcX4
from ensemble.logitboost import BinaryLogitBoost
from ensemble.rotation_forest import RotationForest
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Dictionary with all studied classifiers and their grid search parameters
# Example:
# - output a Random Forets classifier: classifiers['RF']['clf']
# - output the grid search parameters for the Adaboost classifier: classifiers['Ada']['grid_search']

classifiers = {
                'Bag':
                    {'clf' : BaggingClassifier(),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                     },
                'BagET':
                    {'clf' : BaggingClassifier(base_estimator=ExtraTreeClassifier()),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                     },
                'RF':
                    {'clf' : RandomForestClassifier(),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                     },
                'RadP':
                    {'clf' : BaggingClassifier(),
                         'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
                                           'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                           'max_features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            },
                         'random_search': {'n_estimators' : randint(100, 1000),
                                           'max_samples': uniform(0.2, 1),
                                           'max_features': uniform(0.3, 1)}
                    },
                'RadPET':
                    {'clf' : BaggingClassifier(base_estimator=ExtraTreeClassifier()),
                         'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
                                           'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                           'max_features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                            },
                         'random_search': {'n_estimators' : randint(100, 1000),
                                           'max_samples': uniform(0.2, 1),
                                           'max_features': uniform(0.3, 1)}
                    },
                'Ad':
                    {'clf' : AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4)),
                         'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                         'random_search': {'n_estimators' : randint(100, 1000)}
                    },
                'AdSt':
                    {'clf' : AdaBoostClassifier(),
                         'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                         'random_search': {'n_estimators' : randint(100, 1000)}
                    },
                'AdET':
                    {'clf' : AdaBoostClassifier(base_estimator=ExtraTreeClassifier()),
                         'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                         'random_search': {'n_estimators' : randint(100, 1000)}
                    },
                'Vad':
                    {'clf' : BinaryVadaboost(),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
                                     'lbda' : np.arange(0.05, 1.05, 0.05)},
                     'random_search': {'n_estimators' : randint(100, 1000),
                                       'lbda': uniform(0.05, 1)}
                    },
                'VadET':
                    {'clf' : BinaryVadaboost(base_estimator=ExtraTreeClassifier()),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100),
                                     'lbda' : np.arange(0.05, 1.05, 0.05)},
                     'random_search': {'n_estimators' : randint(100, 1000),
                                       'lbda': uniform(0.1, 1)}
                    },
                'Swt':
                    {'clf' : ClassSwitching(),
                    'grid_search': {'n_estimators' : np.arange(100, 600, 100),
                                   'p_switch' : np.arange(0.1, 0.4, 0.05)},
                    'random_search': {'n_estimators' : randint(100, 600, 100),
                                     'p_switch' : uniform(0.1, 0.4)}
                    },
                'SwtET':
                    {'clf' : ClassSwitching(base_estimator=ExtraTreeClassifier()),
                    'grid_search': {'n_estimators' : np.arange(100, 600, 100),
                                   'p_switch' : np.arange(0.1, 0.4, 0.05)},
                    'random_search': {'n_estimators' : randint(100, 600, 100),
                                     'p_switch' : uniform(0.1, 0.4)}
                    },
                'ArcX4':
                    {'clf' : ArcX4(),#n_jobs=4),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                    },
                'ArcX4ET':
                    {'clf' : ArcX4(base_estimator=DecisionTreeClassifier()),#, n_jobs=4),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                    },
                'LogB':
                    {'clf' : BinaryLogitBoost(),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                     'random_search': {'n_estimators' : randint(100, 1000)}
                     },
                'Rot':
                    {'clf' : RotationForest(K=3),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                    },
                'RotET':
                    {'clf' : RotationForest(K=3),
                     'grid_search': {'n_estimators' : np.arange(100, 1100, 100)},
                    },
                'Rotb':
                    {'clf' : RotationForest(K=3),
                     'grid_search': {'n_estimators' : np.arange(10, 110, 10),
                                     'base_estimator' : [AdaBoostClassifier(n_estimators=5),
                                                         AdaBoostClassifier(n_estimators=10)]},
                    },
                'RotbET':
                    {'clf' : RotationForest(K=3),
                     'grid_search': {'n_estimators' : np.arange(10, 110, 10),
                                     'base_estimator' :
                                         [AdaBoostClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=5),
                                          AdaBoostClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=10)]
                                     },
                    },
                'CART':
                    {'clf' : DecisionTreeClassifier()},
                'ET':
                    {'clf' : ExtraTreeClassifier()}
                }

def load_classifiers(classifier_name):
    """
    :param classifier_name: string that represents the classifier's nickname
    :return: load the class object of the classifier
    """
    if classifier_name not in classifiers.keys():
        raise ValueError("Classifier's name not recognized")
    else:
        return classifiers[classifier_name]['clf']

def load_classifiers_names():
    """
    :param None
    :return: all the names of the classifiers available
    """
    return classifiers.keys()

def load_grid_parameters(classifier_name):
    """
    :param classifier_name: string that represents the classifier's nickname
    :return: Grid Search parameters used to tune the classifier
    """
    return classifiers[classifier_name]['grid_search']

def load_random_parameters(classifier_name):
    """
    :param classifier_name: string that represents the classifier's nickname
    :return: Random Grid Search parameters used to tune the classifier
    """
    return classifiers[classifier_name]['random_search'], classifiers[classifier_name]['n_random_iter']

def number_iterations(dictionary):
    for classifier in dictionary.keys():
        if 'grid_search' in dictionary[classifier].keys():
            grid_search = dictionary[classifier]['grid_search']
            result = [len(grid_search[parameter]) for parameter in grid_search.keys()]
            dictionary[classifier]['n_random_iter'] = np.product(result)
    return dictionary


