import os
def generate_sequence(sequence):
    # 把 (a, b) 對轉換為 "a+b" 的形式
    str_seq = [f"{a}+{b}" for a, b in sequence]

    # 用 "-" 連接所有的 "a+b"
    result = "-".join(str_seq)
    
    return result
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
        # 'xgb',
        'dts',
        # 'rf'
        ]

behav =['sleep','other','meal','out']
rang=[[(0, 1), (1, 5)],[(0, 3), (3, 5)],[(0, 2), (2, 5)],[(0, 2), (1, 5)]]


for item, ra in zip(behav, rang):
    for k in kernel:
        for j in range(2,9):
            for _ in range(5):
                cmds = []
                params_dict['K']=10*j
                params_dict['C']=20*j
                params_dict['kernel']=k
                params_dict['var']=generate_sequence([(0,5)])
                for para_n,para_v in params_dict.items():
                    cmds.append('--'+para_n)
                    cmds.append(str(para_v))
                cmd = ' '.join(cmds)
                # print(cmd)
                run_command(cmd,item)
                cmds = []
                params_dict['var']=generate_sequence(ra)
                for para_n,para_v in params_dict.items():
                    cmds.append('--'+para_n)
                    cmds.append(str(para_v))
                cmd = ' '.join(cmds)
                run_command(cmd,item)
                # print(cmd)
            # import sys
            # sys.exit()
# print(cmd)