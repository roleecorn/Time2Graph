import argparse
import itertools
def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for your program")

    parser.add_argument('--seg_length', type=int, default=5, help='Segment length')
    parser.add_argument('--num_segment', type=int, default=5, help='Number of segments')
    parser.add_argument('--behav', type=str, default='sleep', help='Behavior')
    parser.add_argument('--K', type=int, default=20, help='K')
    parser.add_argument('--C', type=int, default=40, help='C')
    parser.add_argument('--opt_metric', type=str, default='accuracy', help='Optimization metric')
    parser.add_argument('--init', type=int, default=0, help='Init')
    parser.add_argument('--warp', type=int, default=2, help='Warp')
    parser.add_argument('--gpu_enable', type=bool, default=True, help='Enable GPU')
    parser.add_argument('--tflag', type=bool, default=True, help='T flag')
    parser.add_argument('--embed', type=str, default='aggregate', help='Embed')
    parser.add_argument('--cmethod', type=str, default='greedy', help='C method')
    parser.add_argument('--percentile', type=int, default=10, help='Percentile')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--embed_size', type=int, default=16, help='Embed size')
    parser.add_argument('--njobs', type=int, default=5, help='Number of jobs')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--measurement', type=str, default='gdtw', help='Measurement')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha')
    parser.add_argument('--beta', type=float, default=0.05, help='Beta')
    parser.add_argument('--scaled', type=bool, default=True, help='Scaled')
    parser.add_argument('--norm', type=bool, default=False, help='Norm')
    parser.add_argument('--no_global', type=bool, default=True, help='No global')
    parser.add_argument('--multi_graph', type=bool, default=False, help='Multi graph')
    parser.add_argument('--data_size', type=int, default=1, help='Data size')
    parser.add_argument('--kernel', type=str, default='dts', help='Kernel')
    parser.add_argument('--feature', type=str, default='all', help='Feature')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits')

    args = parser.parse_args()

    return args

def opt_clf_para(kernel):
    if kernel=='xgb':
        opt_args= {
                    'max_depth': 16,
                    'learning_rate': 0.2,
                    'scale_pos_weight': 1,
                    'booster': 'gbtree'
                }
    elif (kernel=='dts') or (kernel=='rf'):
        opt_args={
                    'criterion': 'gini',
                    'max_features': 'auto',
                    'max_depth': 10,
                    'min_samples_split': 4,
                    'min_samples_leaf': 3,
                    'class_weight': 'balanced'
                }
    else:
        raise ValueError("please choose a classifier")
    return opt_args

import itertools

def t2g_paras():
    k =  [i*10 for i in range(2,10)]
    k = [10]
    opt_metric = ['f1']
    cmethod =['greedy','cluster']
    percentile =[5,10]
    embed_size = [16,32,64]
    optimizer = ['Adam','Adadelta','Adamax']
    measurement =['gdtw','gw']
    scaled =[True]
    kernel =['dts']
    # 8*2*2*3*3

    for (p1, p2, p3, p4, p5, p6, p7, p8, p9) in itertools.product(
            k, opt_metric, cmethod, percentile, embed_size, optimizer, measurement, scaled, kernel
    ):
        yield {
            'K': p1,
            'C': p1*6,
            'opt_metric': p2,
            'cmethod': p3,
            'percentile': p4,
            'embed_size': p5,
            'optimizer': p6,
            'measurement': p7,
            'scaled': p8,
            'kernel': p9
        }


def clf_paras(kernel):
    class_weight = 'balanced' 
    if kernel == 'lr':
        penalty = ['l1', 'l2']
        c = [pow(5, i) for i in range(-3, 3)]
        intercept_scaling = [pow(5, i) for i in range(-3, 3)]
        for (p1, p2, p3) in itertools.product(penalty, c, intercept_scaling):
            yield {
                'penalty': p1,
                'C': p2,
                'intercept_scaling': p3,
                'class_weight': class_weight
            }
    elif kernel == 'rf' or kernel == 'dts':
        criteria = ['gini', 'entropy']
        max_features = ['auto', 'log2']
        max_depth = [10, 25, 50]
        min_samples_split = [2, 4, 8]
        min_samples_leaf = [1, 3, 5]
        # 2*3*3*3*3
        for (p1, p2, p3, p4, p5) in itertools.product(
                criteria, max_features, max_depth, min_samples_split, min_samples_leaf
        ):
            yield {
                'criterion': p1,
                'max_features': p2,
                'max_depth': p3,
                'min_samples_split': p4,
                'min_samples_leaf': p5,
                'class_weight': class_weight
            }
    elif kernel == 'xgb':
        max_depth = [1, 2, 4, 8, 12, 16]
        learning_rate = [0.1, 0.2, 0.3]
        n_jobs = [5]
        class_weight = [1, 10, 50, 100]
        booster = ['gblinear', 'gbtree', 'dart']
        # 6*3*4*3
        for (p1, p2, p3, p4, p5) in itertools.product(
                max_depth, learning_rate, booster, n_jobs, class_weight
        ):
            yield {
                'max_depth': p1,
                'learning_rate': p2,
                'booster': p3,
                'n_jobs': p4,
                'scale_pos_weight': p5
            }
    elif kernel == 'svm':
        c = [pow(2, i) for i in range(-2, 2)]
        svm_kernel =  ['rbf', 'poly', 'sigmoid']
        for (p1, p2) in itertools.product(c, svm_kernel):
            yield {
                'C': p1,
                'kernel': p2,
                'class_weight': class_weight
                }
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    args = parse_args()
