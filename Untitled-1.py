
import argparse
import warnings
import pandas as pd
import os
os.system("export PYTHONPATH=`readlink -f ./`")
from config import *
from houses import TEST_HOUSE,TRAIN_HOUSE
from archive.load_tepco import load_house_dataset_by_houses,load_house_dataset_by_houses_ex
from time2graph.utils.base_utils import Debugger
from time2graph.core.model_TEPCO import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from time2graph.core.shapelet_embedding import ShapeletEmbedding
from time2graph.core.shapelet_utils import transition_matrixs,__mat2edgelist,graph_embedding,shapelet_distance
from sklearn.preprocessing import minmax_scale
testhouse = [str(i).zfill(3) for i in TEST_HOUSE]
trainhouse = [str(i).zfill(3) for i in TRAIN_HOUSE]


# ## 參數集合


behav='sleep'
class args:
    pass
args.seg_length,args.num_segment =5,5
args.cutpoints=[(0,3),(2,5)]
args.behav=behav
args.dataset = args.behav
args.K,args.C=20,40
args.opt_metric = 'accuracy'
args.init,args.warp=0,2
args.gpu_enable =True
args.tflag=True
args.embed,args.cmethod ='aggregate','greedy'
args.percentile =10
args.batch_size,args.embed_size =16,16
args.njobs=5
args.optimizer ,args.measurement='Adam','gdtw'
args.alpha,args.beta=0.1,0.05
args.scaled,args.norm=True,False
args.no_global = True
args.multi_graph,args.data_size =False,1
args.kernel,args.feature = 'dts','all'
args.n_splits = 5


if args.kernel=='xgb':
    opt_args= {
                'max_depth': 16,
                'learning_rate': 0.2,
                'scale_pos_weight': 1,
                'booster': 'gbtree'
            }
elif (args.kernel=='dts') or (args.kernel=='rf'):
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


# ## load data


x_train, y_train, x_test, y_test,z_train,z_test = load_house_dataset_by_houses_ex(
        TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior=behav)
Debugger.info_print('data shape {}x{}'.format(x_train.shape[0],x_train.shape[1]))
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
                    feature_mode = args.feature,
                    label_all = args.behav,
                    cutpoints=args.cutpoints,
                )


# ### 學習shapelet


cache_dir = '{}/scripts/cache/{}/'.format(module_path, args.dataset)
m.learn__shapelet(X=x_train, Y=y_train,Z=z_train, cache_dir=cache_dir)


assert m.t2g.sembeds is None
for k in range(m.data_size):
    m.data_scaler[k].fit(x_train[:, :, k])
X_scaled = np.zeros(x_train.shape, dtype=np.float)
for k in range(m.data_size):
    X_scaled[:, :, k] = m.data_scaler[k].fit_transform(x_train[:, :, k])
X_scaled = np.zeros(x_train.shape, dtype=np.float)
if args.scaled:
    Debugger.info_print('scaled embedding model...')
    inputx=X_scaled
else:
    Debugger.info_print('unscaled embedding model...')
    inputx=x_train


# ### m.t2g.sembeds 是ShapeletEmbedding


assert m.t2g.shapelets is not None, 'shapelets has not been learnt yet'
m.t2g.sembeds = ShapeletEmbedding(
    seg_length=args.seg_length, tflag=args.tflag, multi_graph=args.multi_graph,
    cache_dir=cache_dir, tanh=False, debug=m.t2g.debug,
    percentile=args.percentile, measurement=args.measurement, mode=args.embed,
    global_flag=args.no_global, 
    **m.t2g.kwargs)



# ### 計算前後段的transition_matrix
# 
# 這邊不太確定y_train == 1還是==0


def shape_norm(tmat,num_shapelet):
    for i in range(num_shapelet):
        norms = np.sum(tmat[0, i, :])
        if norms == 0:
            tmat[k, i, i] = 1.0
        else:
            tmat[k, i, :] /= np.sum(tmat[k, i, :])
    return tmat
transition_set=transition_matrixs(
            time_series_set=x_train[np.argwhere(y_train == 1).reshape(-1), :, :], 
            shapelets=m.t2g.shapelets, seg_length=args.seg_length,
            tflag=args.tflag, multi_graph=args.multi_graph, tanh=False, debug=True,
            init=args.init, warp=args.warp, percentile=args.percentile, threshold=-1,
            measurement=args.measurement, global_flag=args.no_global,
            cutpoints=args.cutpoints
            )


# ## 這裡要計算相鄰的兩個cutpoint間的關係


__cmat_threshold = 1e-2
def cross_matrix(time_series_set, shapelets, seg_length, tflag, multi_graph,
                      percentile, threshold, tanh, debug, init, warp, measurement, 
                      global_flag, cutpoints=[]):
    gcnt=1
    start1,end1=cutpoints[0]
    start2,end2=cutpoints[1]
    num_segment =end1-start1
    start1,end1,start2,end2=start1*seg_length,end1*seg_length,start2*seg_length,end2*seg_length
    num_shapelet = len(shapelets)
    num_time_series = time_series_set.shape[0]
    cmat = np.zeros((1, num_shapelet, num_shapelet), dtype=np.float32)
    sdist1 = shapelet_distance(
        time_series_set=time_series_set[:, start1:end1], shapelets=shapelets, seg_length=seg_length, tflag=tflag,
        tanh=tanh, debug=debug, init=init, warp=warp, measurement=measurement, global_flag=global_flag
    )
    sdist2 = shapelet_distance(
        time_series_set=time_series_set[:, start2:end2], shapelets=shapelets, seg_length=seg_length, tflag=tflag,
        tanh=tanh, debug=debug, init=init, warp=warp, measurement=measurement, global_flag=global_flag
    )
    if percentile is not None:
        dist_threshold = max(np.percentile(sdist1, percentile), np.percentile(sdist2, percentile))
        Debugger.info_print('threshold({}) {}, mean {}'.format(percentile, dist_threshold, (np.mean(sdist1)+np.mean(sdist2))/2))
    else:
        dist_threshold = threshold
        Debugger.info_print('threshold {}, mean {}'.format(dist_threshold, (np.mean(sdist1)+np.mean(sdist2))/2))
    Debugger.info_print(f'{num_time_series}x{num_segment}')
    n_edges = 0
    # 核心計算位置
    for tidx in range(num_time_series):
        for sidx in range(num_segment):
            src_dist = sdist1[tidx, sidx, :]
            dst_dist = sdist2[tidx, sidx, :]
            src_idx = np.argwhere(src_dist <= dist_threshold).reshape(-1)
            dst_idx = np.argwhere(dst_dist <= dist_threshold).reshape(-1)
            if len(src_idx) == 0 or len(dst_idx) == 0:
                continue
            n_edges += len(src_idx) * len(dst_idx)
            src_dist[src_idx] = 1.0 - minmax_scale(src_dist[src_idx])
            dst_dist[dst_idx] = 1.0 - minmax_scale(dst_dist[dst_idx])
            for src in src_idx:
                if multi_graph:
                    cmat[sidx, src, dst_idx] += (src_dist[src] * dst_dist[dst_idx])
                else:
                    cmat[0, src, dst_idx] += (src_dist[src] * dst_dist[dst_idx])
        Debugger.debug_print(
            '{:.2f}% transition matrix computed...'.format(float(tidx + 1) * 100 / num_time_series),
            debug=debug
        )
    Debugger.info_print('{} edges involved in shapelets graph'.format(n_edges))
    cmat[cmat <= __cmat_threshold] = 0.0
    for k in range(gcnt):
        for i in range(num_shapelet):
            norms = np.sum(cmat[k, i, :])
            if norms == 0:
                cmat[k, i, i] = 1.0
            else:
                cmat[k, i, :] /= np.sum(cmat[k, i, :])
    return cmat
cmat = cross_matrix(time_series_set=x_train[np.argwhere(y_train == 0).reshape(-1), :, :], 
            shapelets=m.t2g.shapelets, seg_length=args.seg_length,
            tflag=args.tflag, multi_graph=args.multi_graph, tanh=False, debug=True,
            init=args.init, warp=args.warp, percentile=args.percentile, threshold=-1,
            measurement=args.measurement, global_flag=args.no_global,
            cutpoints=args.cutpoints)


# ## 做embedding，之後要修正


for idx,transition in enumerate(transition_set):
    tmat, sdist, dist_threshold = transition
    tmat = shape_norm(tmat=tmat,num_shapelet=args.K)
    m.t2g.sembeds.dist_threshold = dist_threshold
    m.t2g.sembeds.embeddings.append(
        graph_embedding(
        tmat=tmat, num_shapelet=len(m.t2g.shapelets), embed_size=args.embed_size,
        cache_dir=cache_dir, **m.t2g.sembeds.deepwalk_args)
    )


x = m.extract_features(X=x_train,Z=z_train, init=args.init,mode=args.feature)
max_accu, max_prec, max_recall, max_f1, max_metric = -1, -1, -1, -1, -1
metric_measure = m.return_metric_method(opt_metric=m.t2g.opt_metric)
m.train_classfit(x=x,Y=y_train,Z=z_train,n_splits=5,opt_args=opt_args)


y_pred = m.predict(X=x_test,Z=z_test)[0]


Debugger.dc_print('{}\n{:.2f} positive ratio\nresult: accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}'.format(
        args.cutpoints,float(sum(y_test) / len(y_test)),                                                                           
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))


