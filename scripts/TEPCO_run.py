# -*- coding: utf-8 -*-
import argparse
import warnings
import os
from config import *
from archive.load_tepco import load_house_dataset_by_houses
from time2graph.utils.base_utils import Debugger
from time2graph.core.model import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
MISS_HOUSE=[0,49, 48, 40, 50, 41, 44, 46, 43, 4, 42, 47, 45,75]
TRAIN_HOUSE=[62, 84, 82, 17, 52, 35, 11, 94, 13, 79, 109, 32, 30, 101, 7, 58, 83, 37, 18,
             36, 9, 68, 29, 21, 93, 27, 66, 15, 26, 5, 19, 51, 10, 90, 8, 92, 38, 99, 69,
             67, 56, 63, 80, 59, 74, 108, 91, 54, 71, 61, 98, 70, 34, 105, 72, 77, 104,
             2, 65, 23, 3, 25, 102, 88, 100, 96, 22, 73, 85, 16, 106, 64, 110,31, 86,39,]
TEST_HOUSE=[76, 53, 107, 20, 95, 78, 55, 12, 1, 33, 89, 60, 28, 57, 6, 24, 14, 103, 87]
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
    parser.add_argument('--opt_metric', type=str, default='accuracy', help='which metric to optimize in prediction')
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
    x_train, y_train, x_test, y_test = load_house_dataset_by_houses(
        TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior=args.behav)
    print(x_train.shape)

    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(float(sum(y_train) / len(y_train)),
                                                                         len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(float(sum(y_test) / len(y_test)),
                                                                     len(y_test)))
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
    Debugger.info_print('shapelets_cache={}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
                       module_path, args.dataset, args.cmethod, args.K, args.seg_length)
                   )
    res = np.zeros(4, dtype=np.float32)
    Debugger.info_print('training {}_mixed_model ...'.format(args.dataset))
    cache_dir = '{}/scripts/cache/{}/'.format(module_path, args.dataset)
    if not path.isdir(cache_dir):
        os.mkdir(cache_dir)
    m.fit(X=x_train, Y=y_train, cache_dir=cache_dir, n_splits=args.n_splits)
    if args.cache:
        m.save_model(fpath='{}/scripts/cache/{}_embedding_t2g_model.cache'.format(module_path, args.dataset))
    Debugger.info_print('only predict label not probility')
    y_pred = m.predict(X=x_test)[0]
    
    Debugger.info_print('result: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
    with open('TEPCO_result.csv',mode='a+') as f:
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
            args.behav,
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
