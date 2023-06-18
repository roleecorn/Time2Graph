import os
for i in range(110):
    os.system('python to_tsv.py --house {} --mode train'.format(i))
    os.system('python to_tsv.py --house {} --mode test'.format(i))