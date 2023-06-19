import os
import concurrent.futures

# 定義一個函數來運行命令
def run_command(i):
    os.system('python scripts/run.py --dataset ucr-Power{}   --kernel dts --seg_length 60 --K 30'.format(i))

# 創建一個 ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    # 使用 executor.map 函數來並行地運行指令
    executor.map(run_command, range(100,110))
