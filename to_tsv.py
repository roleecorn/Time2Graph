import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import argparse
MISS_HOUSE=[0,49, 48, 40, 50, 41, 44, 46, 43, 4, 42, 47, 45]

input_path = "./total6_light"
time_windows = 1440  # 一天的分鐘數


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--house', type=int, default=1,help='household to choose')
parser.add_argument('--mode', type=str, default='test', help='test or train')
parser.add_argument('--window', type=int, default=1440,help='time_windows')
args = parser.parse_args()


if args.house in MISS_HOUSE:
    raise ValueError("The provided house value is not valid.")
if args.mode == 'test':
    truedata = 20  # 正樣本數量
    falsedata = 20  # 負樣本數量
elif args.mode == 'train':
    truedata = 50  # 正樣本數量
    falsedata = 50  # 負樣本數量
else:
    raise ValueError('mode must be train ot test')
def load_housedata(houseid:int):
    filename = f"{input_path}/{str(houseid).zfill(3)}_light.csv"
    df= pd.read_csv(filename, usecols=['total_power'],dtype=float)
    df['total_power'].fillna(0, inplace=True)
    n_complete_segments = df.shape[0] // time_windows

    trimmed_data = df['total_power'].values[:n_complete_segments * time_windows]
    split_data = np.array_split(trimmed_data, n_complete_segments)
    scaler = MinMaxScaler()
    normalized_data = [scaler.fit_transform(data.reshape(-1,1)).flatten() for data in split_data]
    return normalized_data

def Xydata(targethouse:int):
    samples = []
    housedata = load_housedata(targethouse)
    n=len(housedata)

    for _ in range(truedata):
        sample = housedata[random.randint(0, n-1)]
        samples.append(sample)

    directory = input_path
    filenames = os.listdir(directory)
    for _ in range(falsedata):
        target = random.sample(filenames,1)[0].split('_')[0]
        target = int(target)
        while target == targethouse: 
            target = random.sample(filenames,1)[0].split('_')[0]
            target = int(target)
        tmp =load_housedata(target)
        n= len(tmp)
        sample = tmp[random.randint(0, n-1)]
        samples.append(sample)
    X = np.stack(samples)
    array_ones = np.ones(truedata)
    array_zeros = np.zeros(falsedata)
    y = np.concatenate((array_ones, array_zeros))
    return X,y

def write_tsv(filename: str, X: np.ndarray, y: np.ndarray):
    with open(filename, 'w') as f:
        for xi, yi in zip(X, y):
            f.write(str(int(yi)) + '\t' + '\t'.join(map(str, xi)) + '\n')

# 使用範例

p='dataset/UCRArchive_2018'
X, y = Xydata(args.house)
filepath = f'{p}/Power{args.house}'
if not os.path.exists(filepath):
    os.makedirs(filepath)
write_tsv(f'{filepath}/Power_{args.mode.upper()}.tsv', X, y)
