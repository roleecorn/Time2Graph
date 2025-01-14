import os
import concurrent.futures

# 定義一個函數來運行命令
params_dict = {
    'K': 20,
    'C': 50,
    'n_splits': 5,
    'num_segment': 3,
    'seg_length': 5,
    'opt_metric': 'recall',
    'embed': 'aggregate',
    'embed_size': 64,
    'warp': 2,
    'kernel': 'dts',
    'percentile': 10,
    'batch_size': 50,
    'scaled': '',
    'cache':'',
}
def run_command(cmds,i):
    # os.system('python scripts/TEPCO_run.py   --seg_length 5 --num_segment 3   --kernel dts --behav {} --embed_size 64'.format(i))
    os.system('python scripts/TEPCO_run.py {} --behav {}'.format(cmds,i))

assign={'sleep':1,'out':2,'meal':3,'other':4}
for i in assign.keys():
    for j in range(1,11):
        cmds = []
        params_dict['K']=10*j
        params_dict['C']=20*j
        for para_n,para_v in params_dict.items():
            cmds.append('--'+para_n)
            cmds.append(str(para_v))
        cmd = ' '.join(cmds)
        run_command(cmd,i)
# print(cmd)