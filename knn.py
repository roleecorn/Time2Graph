from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from houses import TEST_HOUSE,TRAIN_HOUSE
from archive.load_tepco import load_house_dataset_by_houses
import argparse
import warnings
from sktime.classification.interval_based import TimeSeriesForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time
import numpy as np
testhouse = [str(i).zfill(3) for i in TEST_HOUSE]
trainhouse = [str(i).zfill(3) for i in TRAIN_HOUSE]
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--behav', type=str, default='out', 
                    help='which to classify')
parser.add_argument('--k', type=int, default=3, 
                    help='knn n_neighbors')
args = parser.parse_args()

def knn_classifier(x_train, y_train, x_test, params):
    knn = KNeighborsClassifier(**params)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return y_pred

def label_propagation_classifier(x_train, y_train, x_test, params):
    lp = LabelSpreading(**params)
    lp.fit(x_train, y_train)
    y_pred = lp.predict(x_test)
    return y_pred

x_train, y_train, x_test, y_test,z_train,z_test = load_house_dataset_by_houses(trainhouse, testhouse, args.behav)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
z_train = z_train.reshape(-1, 1)
z_test = z_test.reshape(-1, 1)
x_train=np.concatenate((x_train, z_train), axis=1)
x_test=np.concatenate((x_test, z_test), axis=1)
knn_params = {
    'n_neighbors': args.k,
    'weights': 'distance',
    'algorithm': 'ball_tree',
    'leaf_size': 30,
    'p': 2,
    'metric': 'minkowski'
}

lp_params = {
    'kernel': 'knn',
    'gamma': 20,
    'n_neighbors': args.k,
    'alpha': 0.1,
    'max_iter': 30,
    'tol': 0.001
}
start= time()
y_pred_knn = knn_classifier(x_train, y_train, x_test, knn_params)
knntime=time()-start
y_pred_label_propagation = label_propagation_classifier(x_train, y_train, x_test, lp_params)
laptime=time()-start-knntime



with open('knn_result.csv',mode='a+') as f:
    f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n'.format(
        args.behav,
        accuracy_score(y_true=y_test, y_pred=y_pred_knn),
        precision_score(y_true=y_test, y_pred=y_pred_knn),
        recall_score(y_true=y_test, y_pred=y_pred_knn),
        f1_score(y_true=y_test, y_pred=y_pred_knn),
        knntime,
    ))
with open('lap_result.csv',mode='a+') as f:
    f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n'.format(
        args.behav,
        accuracy_score(y_true=y_test, y_pred=y_pred_knn),
        precision_score(y_true=y_test, y_pred=y_pred_knn),
        recall_score(y_true=y_test, y_pred=y_pred_knn),
        f1_score(y_true=y_test, y_pred=y_pred_knn),
        laptime,
    ))