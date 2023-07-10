import os
import concurrent.futures

# 定義一個函數來運行命令
params_dict = {
    'K': 20,
    'C': 50,
    'n_splits': 5,
    'num_segment': 5,
    'seg_length': 5,
    'opt_metric': 'precision',
    # precision
    'embed': 'aggregate',
    'embed_size': 16,
    'warp': 2,
    # 'kernel': 'dts',
    'percentile': 10,
    'batch_size': 50,
    'scaled': '',
    # 'cache':'',
    'feature':'all',
    'resample':""
}
def run_command(cmds,i):
    # os.system('python scripts/TEPCO_run.py   --seg_length 5 --num_segment 3   --kernel dts --behav {} --embed_size 64'.format(i))
    os.system('python scripts/TEPCO_run_ex.py {} --behav {}'.format(cmds,i))

assign={
        # 'all':0,
        'sleep':1,
        'out':2,
        'meal':3,
        'other':4
        }
kernel=[
        'xgb',
        # 'dts',
        # 'rf'
        ]

for i in assign.keys():
    for k in kernel:
        for j in range(3,9):
            cmds = []
            params_dict['K']=10*j
            params_dict['C']=20*j
            params_dict['kernel']=k
            for para_n,para_v in params_dict.items():
                cmds.append('--'+para_n)
                cmds.append(str(para_v))
            cmd = ' '.join(cmds)
            run_command(cmd,i)
            # import sys
            # sys.exit()
# print(cmd)