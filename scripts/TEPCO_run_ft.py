# -*- coding: utf-8 -*-
import argparse
import warnings
import os
from config import *
from houses import TEST_HOUSE,TRAIN_HOUSE
from archive.load_tepco import load_house_dataset_by_houses
from time2graph.utils.base_utils import Debugger
from time2graph.core.model_TEPCO import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
testhouse = [str(i).zfill(3) for i in TEST_HOUSE]
trainhouse = [str(i).zfill(3) for i in TRAIN_HOUSE]
"""
    scripts for running test.
    running command:
        1. set PYTHONPATH environment;
        2. python scripts/run.py **options
        3. option list:
            --dataset, ucr-Earthquakes/WormsTwoClass/Strawberry
            --K, number of shapelets extracted
            --C, number of shapelet candidates
            --n_splits, number of splits in cross-validation
            --num_segment, number of segment a time series is divided into
            --seg_length, segment length
            --njobs, number of threads in parallel
            --data_size, data dimension of time series
            --optimizer, optimizer used in time-aware shapelets learning
            --alpha, penalty parameter of local timing factor
            --beta, penalty parameter of global timing factor
            --init, init index of time series data
            --gpu_enable, bool, whether to use GPU
            --opt_metric, which metric to optimize in prediction
            --cache, whether to dump model to local file
            --embed, which embed strategy to use (aggregate/concatenate)
            --embed_size, embedding size of shapelets
            --warp, warping size in greedy-dtw
            --cmethod, which algorithm to use in candidate generation (cluster/greedy)
            --kernel, specify outer-classifier (default xgboost)
            --percentile, percentile for distance threshold in constructing graph
            --measurement, which distance metric to use (default greedy-dtw)
            --batch_size, batch size in each training step
            --tflag, flag that whether to use timing factors
            --scaled, flag that whether to rescale time series data
            --norm, flag that whether to normalize extracted representations
            --no_global, whether to use global timing factors
"""

if __name__ == '__main__':
    start =time.time()
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--K', type=int, default=100, help='number of shapelets extracted')
    parser.add_argument('--C', type=int, default=200, help='number of shapelet candidates')
    parser.add_argument('--n_splits', type=int, default=5, help='number of splits in cross-validation')
    parser.add_argument('--num_segment', type=int, default=12, help='number of segment a time series is divided into')
    parser.add_argument('--seg_length', type=int, default=30, help='segment length')
    parser.add_argument('--njobs', type=int, default=8, help='number of threads in parallel')
    parser.add_argument('--data_size', type=int, default=1, help='data dimension of time series')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer used in time-aware shapelets learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='penalty parameter of local timing factor')
    parser.add_argument('--beta', type=float, default=0.05, help='penalty parameter of global timing factor')
    parser.add_argument('--init', type=int, default=0, help='init index of time series data')
    parser.add_argument('--gpu_enable', action='store_true', default=False, help='bool, whether to use GPU')
    parser.add_argument('--opt_metric', type=str, default='accuracy', 
                        help='which metric to optimize in prediction,accuracy,precision,recall,f1')
    parser.add_argument('--cache', action='store_true', default=False, help='whether to dump model to local file')
    parser.add_argument('--embed', type=str, default='aggregate',
                        help='which embed strategy to use (aggregate/concatenate)')
    parser.add_argument('--embed_size', type=int, default=256, help='embedding size of shapelets')
    parser.add_argument('--warp', type=int, default=2, help='warping size in greedy-dtw')
    parser.add_argument('--cmethod', type=str, default='greedy',
                        help='which algorithm to use in candidate generation (cluster/greedy)')
    parser.add_argument('--kernel', type=str, default='xgb', help='specify outer-classifier (default xgboost)')
    parser.add_argument('--percentile', type=int, default=10,
                        help='percentile for distance threshold in constructing graph')
    parser.add_argument('--measurement', type=str, default='gdtw',
                        help='which distance metric to use (default greedy-dtw)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size in each training step')
    parser.add_argument('--tflag', action='store_false', default=True, help='flag that whether to use timing factors')
    parser.add_argument('--scaled', action='store_true', default=False,
                        help='flag that whether to rescale time series data')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='flag that whether to normalize extracted representations')
    parser.add_argument('--no_global', action='store_false', default=True,
                        help='whether to use global timing factors')
    parser.add_argument('--multi_graph', action='store_false', default=False,
                        help='whether a multi graph')
    parser.add_argument('--feature', type=str, default='all', 
                        help='what feature want use in classification')
    parser.add_argument('--behav', type=str, default='out', 
                        help='which to classify')
    args = parser.parse_args()
    args.dataset ='001'
    Debugger.info_print('running with {}'.format(args.__dict__))


    # x_train, y_train, x_test, y_test = load_house_dataset_by_name(
    #     fname='001', length=args.seg_length * args.num_segment)
    x_train, y_train, x_test, y_test,z_train,z_test = load_house_dataset_by_houses(
        TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior=args.behav)
    ker = args.kernel
    # x_train, y_train=x_train.reshape(x_train.shape[0], -1),x_test.reshape(x_test.shape[0], -1)
    m = Time2Graph(kernel=args.kernel, K=args.K, C=args.C, seg_length=args.seg_length,
                   opt_metric=args.opt_metric, init=args.init, gpu_enable=args.gpu_enable,
                   warp=args.warp, tflag=args.tflag, mode=args.embed,
                   percentile=args.percentile, candidate_method=args.cmethod,
                   batch_size=args.batch_size, njobs=args.njobs,
                   optimizer=args.optimizer, alpha=args.alpha,
                   beta=args.beta, measurement=args.measurement,
                   representation_size=args.embed_size, data_size=args.data_size,
                   scaled=args.scaled, norm=args.norm, global_flag=args.no_global,
                   multi_graph=args.multi_graph,
                   shapelets_cache='{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
                       module_path, 
                       args.dataset, 
                       args.cmethod, args.K, args.seg_length),
                       feature_mode = args.feature
                   )
    x = m.fm.extract_features(x_train)
    # print(m.fm.clf)
    res = np.zeros(4, dtype=np.float32)
    Debugger.info_print('training {}_mixed_model ...'.format(args.dataset))
    cache_dir = '{}/scripts/cache/{}/'.format(module_path, args.dataset)
    arguments = m.clf_paras(balanced=True)
    print(arguments)
    Y = deepcopy(y_train)
    max_accu, max_prec, max_recall, max_f1, max_metric = -1, -1, -1, -1, -1
    metric_measure = m.return_metric_method(opt_metric=m.t2g.opt_metric)
    for cargs in arguments:
        m.clf.set_params(**cargs)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        tmp = np.zeros(5, dtype=np.float32).reshape(-1)
        measure_vector = [metric_measure, accuracy_score, precision_score, recall_score, f1_score]
        for train_idx, test_idx in skf.split(x, Y):
            m.clf.fit(x[train_idx], Y[train_idx])
            y_pred, y_true = m.clf.predict(x[test_idx]), Y[test_idx]
            for k in range(5):
                tmp[k] += measure_vector[k](y_true=y_true, y_pred=y_pred)
        tmp /= 5
        Debugger.debug_print('cargs tuning: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            tmp[1], tmp[2], tmp[3], tmp[4]
        ), debug=m.verbose)
        if max_metric < tmp[0]:
            max_metric = tmp[0]
            opt_args = cargs
            max_accu, max_prec, max_recall, max_f1 = tmp[1:]
    if m.verbose:
        Debugger.info_print('args {} for clf {}, performance: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            opt_args, m.kernel, max_accu, max_prec, max_recall, max_f1))
    with open(f'{ker}_opt_arg',mode='a+') as ppp:
        ppp.write(str(opt_args))
        ppp.write('\n')
    m.clf.set_params(**opt_args)
    if not path.isdir(cache_dir):
        os.mkdir(cache_dir)
    m.clf.fit(X=x, y=y_train)
    Debugger.info_print('only predict label not probility')
    y_pred = m.clf.predict(X=m.fm.extract_features(x_test))
    
    Debugger.info_print('result: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
    with open('tttttt.csv',mode='a+') as f:
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f},{},{}\n'.format(
            args.behav,
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred),
            time.time()-start,
            args.K,
            args.C,
        ))
