
import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import pandas as pd
import os,time,json,sys
os.system("export PYTHONPATH=`readlink -f ./`")
from config import *
from houses import TEST_HOUSE,TRAIN_HOUSE
from archive.load_tepco import load_house_dataset_by_houses_ex
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.base_utils import Debugger
from time2graph.core.model_TEPCO import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from time2graph.core.shapelet_embedding import ShapeletEmbedding
from time2graph.core.shapelet_utils import transition_matrixs,__mat2edgelist,graph_embedding
from time2graph.core.shapelet_utils import shapelet_distance,cross_graph_embedding
from sklearn.preprocessing import minmax_scale

from cross_matrix import cross_matrix,shape_norm,combine_mat
from t2garg import parse_args,opt_clf_para,t2g_paras,clf_paras
testhouse = [str(i).zfill(3) for i in TEST_HOUSE]
trainhouse = [str(i).zfill(3) for i in TRAIN_HOUSE]

f1max=0
bestarg={},{}
def run(args,opt_args,T2Gidx):
    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        if dataset =='Earthquakes':
            args.seg_length= 24
            args.num_segment=21
            args.cutpoints =[(0,11),(10,21)]
        elif dataset =='Strawberry':
            args.seg_length= 15
            args.num_segment= 15
            args.cutpoints =[(0,8),(7,15)]
        elif dataset == 'WormsTwoClass':
            args.seg_length= 30
            args.num_segment= 29
            args.cutpoints =[(0,15),(14,29)]
        else:
            raise ValueError('not a ucr dataset')
        
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=args.seg_length * args.num_segment)
        z_train,z_test = None,None
    else:
        args.seg_length= 5
        args.num_segment= 5
        args.cutpoints =[(0,3),(3,5)]
        x_train, y_train, x_test, y_test,z_train,z_test = load_house_dataset_by_houses_ex(
                TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior=behav)


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
    x = m.fm.extract_features(samples=x_train)
    y = m.fm.extract_features(samples=x_test)
    kers =['dts','rf','xgb']
    word = ""
    for k in kers:
        m.kernel=k
        for ar_g in clf_paras(kernel=k):
            m.clf=m.clf__()
            # m.train_classfit(x=x,Y=y_train,Z=z_train,n_splits=5,opt_args=ar_g)
            m.clf.set_params(**ar_g)
            m.clf.fit(x, y_train)
            y_pred = m.clf.predict(y)
            accu=accuracy_score(y_true=y_test, y_pred=y_pred)
            prec =precision_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred)
            global f1max,bestarg
            if f1>f1max:
                f1max=f1
                bestarg =(args,ar_g)
                Debugger.info_print('bestarg at f1 = {}'.format(f1))
                time.sleep(1)
                Debugger.info_print(str(args.__dict__))
                time.sleep(1)
                Debugger.info_print(str(ar_g))
            word+='ker:{}\t accu {:.4f}\t prec {:.4f}\t recall {:.4f}\t f1 {:.4f}\n'.format(                                                                          
                    k,accu,prec,recall,f1
                )
            if len(word)>1000:
                Debugger.info_print(word)
                word=""
    Debugger.dc_print(word)      
    return

if __name__ =="__main__":
    results_df = pd.DataFrame(columns=['id','behav', 'accu', 'prec', 'recall', 'f1'])
    params_dict ={}
    T2Gidx = 1
    start =time.time()
    args = parse_args()
    datas_ = ['sleep','out','other','meal','ucr-Earthquakes','ucr-Strawberry','ucr-WormsTwoClass']
    behav='sleep'
    args.dataset=behav
    args.behav = behav
    paras = t2g_paras()
    opt_args = opt_clf_para(args.kernel)
    try :
        run(args=args,opt_args=opt_args,T2Gidx=T2Gidx)
    except KeyboardInterrupt:
        pass
    print(time.time()-start)
    Debugger.dc_print('End')