import os
import concurrent.futures

# 定義一個函數來運行命令
def run_command(i):
    os.system('python to_tsv.py --house {} --mode train'.format(i))
    os.system('python to_tsv.py --house {} --mode test'.format(i))

# 創建一個 ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # 使用 executor.map 函數來並行地運行指令
    executor.map(run_command, range(110))