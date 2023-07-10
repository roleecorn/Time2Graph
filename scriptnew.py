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
    'feature':'all'
}
def run_command(cmds):
    # os.system('python scripts/run.py --dataset ucr-WormsTwoClass --K 400 --C 50 --num_segment 30 --seg_length 30 --percentile 5 --var  {}'.format(cmds))
    # os.system('python scripts/run.py --dataset ucr-Earthquakes --K 800 --C 50 --num_segment 21 --seg_length 24 --percentile 5 --var  {}'.format(cmds))
    os.system('python scripts/run.py --dataset ucr-Strawberry --K 800 --C 50 --num_segment 15 --seg_length 15 --percentile 10 --embed aggregate --var  {}'.format(cmds))
    # os.system

def generate_sequence(sequence):
    # 把 (a, b) 對轉換為 "a+b" 的形式
    str_seq = [f"{a}+{b}" for a, b in sequence]

    # 用 "-" 連接所有的 "a+b"
    result = "-".join(str_seq)
    
    return result

from random import randint

for _ in range(10):
    # seq= generate_sequence([(0,randint(3,15)),(randint(3,14),15)])
    seq = generate_sequence([(0, 10), (7, 15)])
    run_command(seq)
