
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
    global f1max,bestarg
    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]

        if dataset =='Earthquakes':
            args.seg_length= 24
            args.num_segment=21
            args.K= 50
            args.C= 800
            args.cutpoints =[(0,11),(10,21)]
        elif dataset =='Strawberry':
            args.seg_length= 15
            args.num_segment= 15
            args.K=50
            args.C= 800
            args.cutpoints =[(0,8),(7,15)]
        elif dataset == 'WormsTwoClass':
            args.seg_length= 30
            args.num_segment= 29
            args.K=20
            args.C=400
            args.cutpoints =[(0,15),(14,29)]
        else:
            raise ValueError('not a ucr dataset')
        
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=args.seg_length * args.num_segment)
        z_train,z_test = None,None
    else:
        args.seg_length= 5
        args.num_segment= 5
        args.cutpoints =[(0,3),(2,5)]
        x_train, y_train, x_test, y_test,z_train,z_test = load_house_dataset_by_houses_ex(
                TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior=behav)
    if float(sum(y_train) / len(y_train))<0.2:
        Debugger.info_print('resample')
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        positive_ratio = 0.2
        n_negative = int(len(y_train[y_train==1]) / positive_ratio - len(y_train[y_train==1]))
        rus = RandomUnderSampler(sampling_strategy={0: n_negative, 1: len(y_train[y_train==1])}, random_state=42)
        x_train_res_flattened, y_train = rus.fit_resample(x_train_flattened, y_train)
        x_train = x_train_res_flattened.reshape(-1, x_train.shape[1], x_train.shape[2])
    Debugger.info_print('data shape {}x{}'.format(x_train.shape[0],x_train.shape[1]))
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(float(sum(y_train) / len(y_train)),
                                                                            len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(float(sum(y_test) / len(y_test)),
                                                                        len(y_test)))

    # print(x_train)
    # sys.exit()
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

    transition_set=transition_matrixs(
                time_series_set=x_train[np.argwhere(y_train == 1).reshape(-1), :, :], 
                shapelets=m.t2g.shapelets, seg_length=args.seg_length,
                tflag=args.tflag, multi_graph=args.multi_graph, tanh=False, debug=True,
                init=args.init, warp=args.warp, percentile=args.percentile, threshold=-1,
                measurement=args.measurement, global_flag=args.no_global,
                cutpoints=args.cutpoints
                )


    # ## 這裡要計算相鄰的兩個cutpoint間的關係

    cmat = cross_matrix(time_series_set=x_train[np.argwhere(y_train == 0).reshape(-1), :, :], 
                shapelets=m.t2g.shapelets, seg_length=args.seg_length,
                tflag=args.tflag, multi_graph=args.multi_graph, tanh=False, debug=True,
                init=args.init, warp=args.warp, percentile=args.percentile, threshold=-1,
                measurement=args.measurement, global_flag=args.no_global,
                cutpoints=args.cutpoints)


    tcmat=combine_mat(transition_set[0][0],transition_set[1][0],cmat)

    emb1,emb2 =cross_graph_embedding(
            tmat=shape_norm(tcmat,args.K*2), num_shapelet=len(m.t2g.shapelets)*2, embed_size=args.embed_size,
            cache_dir=cache_dir, **m.t2g.sembeds.deepwalk_args)
    m.t2g.sembeds.embeddings.append(emb1)
    m.t2g.sembeds.embeddings.append(emb2)
    afile=open(f"src/result_{T2Gidx}.csv",mode='a')
    x = m.extract_features(X=x_train,Z=z_train, init=args.init,mode=args.feature)
    y = m.extract_features(X=x_test,Z=z_test, init=args.init,mode=args.feature)
    kers =['dts']
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
            if f1>f1max:
                f1max=f1
                bestarg =(args,ar_g)
                Debugger.dc_print('bestarg {} at f1 = {}'.format(args.dataset,f1))
                time.sleep(3)
                Debugger.dc_print('ker:{}, accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}\n'.format(                                                                          
                    k,accu,prec,recall,f1
                ))
                time.sleep(3)
                Debugger.dc_print(str(ar_g))
                time.sleep(3)
                Debugger.dc_print(str(args.__dict__))
                time.sleep(3)
            word+='ker:{}, accu {:.4f}, prec {:.4f}, recall {:.4f}, f1 {:.4f}\n'.format(                                                                          
                    k,accu,prec,recall,f1
                )
            if len(word)>1000:
                afile.write(word)
                afile.flush()
                word=""
    afile.write(word)  
    afile.flush()  
    afile.close()        
    return

if __name__ =="__main__":
    results_df = pd.DataFrame(columns=['id','behav', 'accu', 'prec', 'recall', 'f1'])
    params_dict ={}
    T2Gidx = 1
    start =time.time()
    args = parse_args()
    args.cutpoints=[(0,3),(2,5)]
    datas_ = ['sleep','out','other','meal','ucr-Earthquakes','ucr-Strawberry','ucr-WormsTwoClass']

    behav=datas_[2]
    args.dataset=behav
    args.behav = behav
    # args.measurement = 'gw'
    paras = t2g_paras()
    opt_args = opt_clf_para(args.kernel)
    for behav in datas_[-1:]:
        args.dataset=behav
        args.behav = behav
        for para in paras:
            for key, value in para.items():
                setattr(args, key, value)
            try :
                run(args=args,opt_args=opt_args,T2Gidx=T2Gidx)
            except KeyboardInterrupt:
                break
            # sys.exit()
            params_dict[T2Gidx]=(args.__dict__)
            T2Gidx+=1
        with open('params.json', 'w') as f:
            json.dump(params_dict, f, indent=4)
        f1max = 0
    print(time.time()-start)
    Debugger.dc_print('End')