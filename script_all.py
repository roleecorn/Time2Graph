import os
def run_command(i,k):
    os.system('python knn.py  --behav {} --k {}'.format(i,k))

assign={'sleep':1,'out':2,'meal':3,'other':4}
for i in assign.keys():
    for j in range(3,10):
        run_command(i,j)
# print(cmd)