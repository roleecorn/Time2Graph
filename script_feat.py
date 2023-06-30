import os
import concurrent.futures

def run_command(behav,ker):
    # os.system('python scripts/TEPCO_run.py   --seg_length 5 --num_segment 3   --kernel dts --behav {} --embed_size 64'.format(i))
    os.system('python scripts/TEPCO_run_ft.py --behav {} --seg_length 15 --num_segment 1 --kernel {}'.format(behav,ker))

assign={'sleep':1,'out':2,'meal':3,'other':4}
kernel=['svm','dts','rf']
for i in assign.keys():
    for j in kernel:
        run_command(i,j)
# print(cmd)