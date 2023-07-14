import numpy as np
from time2graph.core.shapelet_utils import shapelet_distance 
from config import *
from sklearn.preprocessing import minmax_scale


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
    return cmat


def shape_norm(tmat,num_shapelet):
    for i in range(num_shapelet):
        norms = np.sum(tmat[0, i, :])
        if norms == 0:
            tmat[0, i, i] = 1.0
        else:
            tmat[0, i, :] /= np.sum(tmat[0, i, :])
    return tmat


def combine_mat(tmat1, tmat2, cmat):
    """
    combine three transition matrix tmat1, tmat2, cmat
    """

    shapelet_num = tmat1.shape[1]
    combined_mat = np.zeros((1, shapelet_num*2, shapelet_num*2))

    # Assign the transition of shapelets from tmat1 and tmat2 to the combined matrix
    combined_mat[0, :shapelet_num, :shapelet_num] = tmat1[0]
    combined_mat[0, shapelet_num:, shapelet_num:] = tmat2[0]

    # Use cmat to fill the transition from first half to second half
    combined_mat[0, :shapelet_num, shapelet_num:] = cmat[0]
    # Use transpose of cmat to fill the transition from second half to first half
    combined_mat[0, shapelet_num:, :shapelet_num] = cmat[0].T

    return combined_mat

