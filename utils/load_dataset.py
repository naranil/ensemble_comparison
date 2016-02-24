"""
Functions to load the datasets used for binary classification
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Use those functions to call databases.
This code is given only to show the reader how we construct our databases and store them in a HDFS file. You can
download all the datasets on : https://s3-eu-west-1.amazonaws.com/ensemble-comparison-data/datasets.h5
----------
UCI machine learning repository: http://archive.ics.uci.edu/ml/
"""

import pandas as pd
import numpy as np
import os
import re

from sklearn.preprocessing import LabelBinarizer

# Dataset dictionary. Add a data directory with the following datasets. Otherwise use the url paramater: direct link
# to a dataset.
datasets = {
            'basehock': {'path' : ['/data/basehock/data.csv', '/data/basehock/target.csv']},

            'breast_cancer' : { 'path' : '/data/breast-cancer/breast-cancer-wisconsin.data',
                                'url' : 'breast-cancer-wisconsin/breast-cancer-wisconsin.data'},

            'cleve' : { 'path' : '/data/cleve/processed.cleveland.data',
                        'url' : 'heart-disease/processed.cleveland.data'},

            'colon' : {'path' : ['/data/colon/data.csv', '/data/colon/target.csv']},

            'ionosphere' : { 'path' : '/data/ionosphere/ionosphere.data',
                             'url' : 'ionosphere/ionosphere.data'},

            'leukemia' : {'path' : ['/data/leukemia/leukemia_train.csv', '/data/leukemia/leukemia_test.csv']},

            'madelon' : { 'path' : ['/data/madelon/madelon_train.data', '/data/madelon/madelon_train.labels'],
                          'url' : ['madelon/MADELON/madelon_train.data', 'madelon/MADELON/madelon_train.labels']},

            'musk' : {'path' : ['/data/musk/clean1.data', '/data/musk/clean2.data']},

            'ovarian' : {'path' : '/data/ovarian/ovarian.txt'},

            'parkinson' : { 'path' : '/data/parkinson/parkinsons.data',
                            'url' : 'parkinsons/parkinsons.data'},

            'pima' : { 'path' : '/data/pima/pima-indians-diabetes.data',
                       'url' : 'pima-indians-diabetes/pima-indians-diabetes.data'},

            'pcmac' : {'path' : ['/data/pcmac/data.csv', '/data/pcmac/target.csv']},

            'promoters' : { 'path' : '/data/promoters/promoters.data',
                            'url' : 'molecular-biology/promoter-gene-sequences/promoters.data'},
            'relathe' : {'path' : '/data/relathe/relathe.csv'},

            'smk_can' : {'path' : ['/data/smk-can/data.csv', '/data/smk-can/target.csv']},

            'spam' : { 'path' : '/data/spam/spambase.data',
                       'url' : 'spambase/spambase.data'},

            'spect' : { 'path' : ['/data/spect/SPECT.train', '/data/spect/SPECT.test'],
                        'url' : ['spect/SPECT.train', 'spect/SPECT.test']},

            'wdbc' : { 'path' : '/data/wdbc/wdbc.data',
                       'url' : 'breast-cancer-wisconsin/wdbc.data'},

            'wpbc' : { 'path' : '/data/wpbc/wpbc.data',
                       'url' : 'breast-cancer-wisconsin/wpbc.data'}
            }

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
#absolute_path = os.path.dirname(__file__)
absolute_path = ".."

def load_datasets_names():
    """
    :return: the names of all datasets used.
    """
    return datasets.keys()

def load_big_datasets_names():
    """
    :return: the names of the big datasets. (more than 1000 features)
    """
    return ['basehock', 'colon', 'leukemia', 'ovarian', 'pcmac', 'relathe', 'smk_can']

def load_small_datasets_names():
    """
    :return: the names of the big datasets. (more than 1000 features)
    """
    big_datasets = load_big_datasets_names()
    all_datasets = load_datasets_names()
    return [item for item in all_datasets if item not in big_datasets]

def load_datasets(dataset_name):
    """
    :param dataset_name: dataset name (string)
    :return: (data: np array, target: np array)
    """
    name = re.sub('-', '_', dataset_name)
    return globals()['load_' + name]()

def load_basehock():
    path = datasets['basehock']['path']
    X = pd.read_csv(absolute_path + path[0], header=None)
    y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_breast_cancer(url=False):
    path = datasets['breast_cancer']['path']
    url_path = datasets['breast_cancer']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    data = data[data[6] != '?']
    X = data.iloc[:, 1:10]
    y = data[10]
    y = y.replace(2, 0)
    y = y.replace(4, 1)
    return X.astype(np.float), y

def load_cleve(url=False):
    path = datasets['cleve']['path']
    url_path = datasets['cleve']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    data = data[data[11] != '?']
    data = data[data[12] != '?']
    X = data.iloc[:, :13]
    y = data[13]
    y = y.replace([1, 2, 3, 4], 1)
    return X.astype(np.float), y

def load_colon():
    path = datasets['colon']['path']
    X = pd.read_csv(absolute_path + path[0], header=None)
    y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_ionosphere(url=False):
    path = datasets['ionosphere']['path']
    url_path = datasets['ionosphere']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    X = data.iloc[:, :34]
    y = data[34]
    y = y.replace('g', 1)
    y = y.replace('b', 0)
    return X.astype(np.float), y

def load_leukemia():
    path = datasets['leukemia']['path']
    data_train = pd.read_csv(absolute_path + path[0], header=None)
    data_test = pd.read_csv(absolute_path + path[1], header=None)
    data = pd.concat([data_train, data_test], axis=0)
    X = data.iloc[:, :7129]
    y = data[7129]
    return X.astype(np.float), y

def load_madelon(url=False):
    path = datasets['madelon']['path']
    url_path = datasets['madelon']['url']
    if url:
        X = pd.read_csv(UCI_URL + url_path[0], sep=" ", header=None)
        y = pd.read_csv(UCI_URL + url_path[1], header=None)
    else:
        X = pd.read_csv(absolute_path + path[0], sep=" ", header=None)
        y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(-1, 0)
    return X.iloc[:, :500].astype(np.float), y[0]

def load_musk():
    path = datasets['musk']['path']
    data1 = pd.read_csv(absolute_path + path[0], header=None)
    X = data1.iloc[:, 2:168]
    y = data1[168]
    return X.astype(np.float), y

def load_ovarian():
    path = datasets['ovarian']['path']
    data = pd.read_csv(absolute_path + path, sep="\t", header=None)
    X = data.iloc[:, :1536]
    y = data[1536]
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y

def load_parkinson(url=False):
    path = datasets['parkinson']['path']
    url_path = datasets['parkinson']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path)
    else:
        data = pd.read_csv(absolute_path + path)
    X = data.drop(['status', 'name'], axis=1)
    y = data['status']
    return X.astype(np.float), y

def load_pima(url=False):
    path = datasets['pima']['path']
    url_path = datasets['pima']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    X = data.iloc[:, :8]
    y = data[8]
    return X.astype(np.float), y

def load_pcmac():
    path = datasets['pcmac']['path']
    X = pd.read_csv(absolute_path + path[0], header=None)
    y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_promoters(url=False):
    path = datasets['promoters']['path']
    url_path = datasets['promoters']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    features = data[2].str.replace('\t', '').values
    features = np.array([list(x) for x in features])
    lb = LabelBinarizer()
    lb.fit(['a', 't', 'c', 'g'])
    features = tuple([lb.transform(features[:, i]) for i in range(features.shape[1])])
    X = np.concatenate(features, axis=1)
    y = data[0]
    y = y.replace('+', 1)
    y = y.replace('-', 0)
    y = np.ravel(y)
    return pd.DataFrame(X.astype(np.float)), pd.Series(y)

def load_relathe():
    path = datasets['relathe']['path']
    data = pd.read_csv(absolute_path + path, header=None)
    X = data.iloc[:, :4322]
    y = data[4322]
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y

def load_smk_can():
    path = datasets['smk_can']['path']
    X = pd.read_csv(absolute_path + path[0], header=None)
    y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_spam(url=False):
    path = datasets['spam']['path']
    url_path = datasets['spam']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    X = data.iloc[:, :57]
    y = data[57]
    return X.astype(np.float), y

def load_spect(url=False):
    path = datasets['spect']['path']
    url_path = datasets['spect']['url']
    if url:
        train = pd.read_csv(UCI_URL + url_path[0], header=None)
        test = pd.read_csv(UCI_URL + url_path[1], header=None)
    else:
        train = pd.read_csv(absolute_path + path[0], header=None)
        test = pd.read_csv(absolute_path + path[1], header=None)
    data = pd.concat([train, test], axis=0)
    X = data.iloc[:, :22]
    y = data[22]
    return X.astype(np.float), y

def load_wdbc(url=False):
    path = datasets['wdbc']['path']
    url_path = datasets['wdbc']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    X = data.iloc[:, 2:]
    y = data[1]
    y = y.replace('B', 0)
    y = y.replace('M', 1)
    return X.astype(np.float), y

def load_wpbc(url=False):
    path = datasets['wpbc']['path']
    url_path = datasets['wpbc']['url']
    if url:
        data = pd.read_csv(UCI_URL + url_path, header=None)
    else:
        data = pd.read_csv(absolute_path + path, header=None)
    data = data[data[34].values != '?']
    X = data.iloc[:, 2:]
    y = data[1]
    y = y.replace('N', 0)
    y = y.replace('R', 1)
    return X.astype(np.float), y

if __name__ == '__main__':
    # Change this value to generate the data file
    create_h5_data_file = False

    if create_h5_data_file:
        names =  load_datasets_names()
        datasets = pd.HDFStore('datasets.h5')
        for name in names:
            X, y = load_datasets(name)
            datasets[name + '_target'] = y
            datasets[name + '_data'] = X

        datasets.close()












