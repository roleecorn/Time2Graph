import pandas as pd
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
from houses import TEST_HOUSE,TRAIN_HOUSE
from assign_rule import assign
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
testhouse = [str(i).zfill(3) for i in TEST_HOUSE]
trainhouse = [str(i).zfill(3) for i in TRAIN_HOUSE]

if np.__version__ >= '1.20.0':
    np_float = float
    np_int =int
else:
    np_float = np.float
    np_int = np.int

def to_25(x_train):
    for i in range(x_train.shape[0]):
        if i == 0:
            # 处理第一个样本
            x_train[i][-5:] = x_train[i+1][5:10]
        elif i == x_train.shape[0] - 1:
            # 处理最后一个样本
            x_train[i][:5] = x_train[i-1][10:15]
        else:
            # 处理其他样本
            x_train[i][-5:] = x_train[i+1][5:10]
            x_train[i][:5] = x_train[i-1][10:15]
    return x_train
def load_house_dataset_by_houses(TRAIN_HOUSE, TEST_HOUSE, assign_behavior):
    """
    load TEPCO dataset given household number.
    :param TRAIN_HOUSE:
        list of training household ids
    :param TEST_HOUSE:
        list of testing household ids
    :param length:
        time series length that want to load in.
    :return:
        x_train, y_train_return, x_test, y_test_return, z_train, z_test
    """

    dir_path = '{}/TEPCO_data'.format(module_path)

    x_train, y_train, x_test, y_test = [], [], [], []
    z_train, z_test = [], []  # New lists to store the time information

    # Load the training data
    mapping = assign[assign_behavior]
    for fname in TRAIN_HOUSE:
        labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
                             usecols=['home(sleep)','home(other)','home(meal)','out', 'time'])  # Include 'time'
        labels['behavior'] = labels.apply(mapping, axis=1)
        # Convert 'time' to datetime and extract the hour
        labels['time'] = pd.to_datetime(labels['time']).dt.hour
        datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
        x = datas.values.reshape(-1, 15, 1).astype(np_float)
        y = labels['behavior'].values.astype(np_int)
        z = labels['time'].values  # Extract the 'time' values
        x_train.append(x)
        y_train.append(y)
        z_train.append(z)  # Append the time information

    # Load the testing data
    for fname in TEST_HOUSE:
        labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
                             usecols=['home(sleep)','home(other)','home(meal)','out', 'time'])  # Include 'time'
        labels['behavior'] = labels.apply(mapping, axis=1)
        # Convert 'time' to datetime and extract the hour
        labels['time'] = pd.to_datetime(labels['time']).dt.hour / 24
        datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
        x = datas.values.reshape(-1, 15, 1).astype(np_float)
        y = labels['behavior'].values.astype(np_int)
        z = labels['time'].values  # Extract the 'time' values
        x_test.append(x)
        y_test.append(y)
        z_test.append(z)  # Append the time information

    # Convert list of arrays to one large array
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    z_train = np.concatenate(z_train, axis=0)  # Concatenate the time information
    z_test = np.concatenate(z_test, axis=0)  # Concatenate the time information

    # Convert labels to start from 0
    lbs = np.unique(np.concatenate([y_train, y_test], axis=0))
    y_train_return, y_test_return = np.copy(y_train), np.copy(y_test)
    for idx, val in enumerate(lbs):
        y_train_return[y_train == val] = idx
        y_test_return[y_test == val] = idx

    Debugger.info_print('usr_dataset four return shape \n{}\n{}\n{}\n{}'.format( 
        x_train.shape,y_train_return.shape,x_test.shape,y_test_return.shape))
        
    return x_train, y_train_return, x_test, y_test_return,z_train,z_test 
def load_house_dataset_by_houses_ex(TRAIN_HOUSE, TEST_HOUSE, assign_behavior):
    """
    load TEPCO dataset given household number.
    :param TRAIN_HOUSE:
        list of training household ids
    :param TEST_HOUSE:
        list of testing household ids
    :param length:
        time series length that want to load in.
    :return:
        x_train, y_train_return, x_test, y_test_return, z_train, z_test
    """

    dir_path = '{}/TEPCO_data'.format(module_path)

    x_train, y_train, x_test, y_test = [], [], [], []
    z_train, z_test = [], []  # New lists to store the time information

    # Load the training data
    mapping = assign[assign_behavior]
    for fname in TRAIN_HOUSE:
        labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
                             usecols=['home(sleep)','home(other)','home(meal)','out', 'time'])  # Include 'time'
        labels['behavior'] = labels.apply(mapping, axis=1)
        # Convert 'time' to datetime and extract the hour
        labels['time'] = pd.to_datetime(labels['time']).dt.hour
        datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
        # datas = pd.concat([datas[15:], datas[:15]])
        x = datas.values.reshape(-1, 15, 1).astype(np_float)
        y = labels['behavior'].values.astype(np_int)
        z = labels['time'].values  # Extract the 'time' values

        # Create overlapping sequences with 5 minutes before and after
        x_padded = np.pad(x, ((0, 0), (5, 5), (0, 0)), mode='constant', constant_values=0)
        x_train.append(x_padded)
        y_train.append(y)
        z_train.append(z)

    # Load the testing data
    for fname in TEST_HOUSE:
        labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
                             usecols=['home(sleep)','home(other)','home(meal)','out', 'time'])  # Include 'time'
        labels['behavior'] = labels.apply(mapping, axis=1)
        # Convert 'time' to datetime and extract the hour
        labels['time'] = pd.to_datetime(labels['time']).dt.hour / 24
        datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
        x = datas.values.reshape(-1, 15, 1).astype(np_float)
        y = labels['behavior'].values.astype(np_int)
        z = labels['time'].values  # Extract the 'time' values

        # Create overlapping sequences with 5 minutes before and after
        x_padded = np.pad(x, ((0, 0), (5, 5), (0, 0)), mode='constant', constant_values=0)
        x_test.append(x_padded)
        y_test.append(y)
        z_test.append(z)

    # Convert list of arrays to one large array
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    z_train = np.concatenate(z_train, axis=0)  # Concatenate the time information
    z_test = np.concatenate(z_test, axis=0)  # Concatenate the time information

    # Convert labels to start from 0
    lbs = np.unique(np.concatenate([y_train, y_test], axis=0))
    y_train_return, y_test_return = np.copy(y_train), np.copy(y_test)
    for idx, val in enumerate(lbs):
        y_train_return[y_train == val] = idx
        y_test_return[y_test == val] = idx
    x_train = to_25(x_train)
    x_test = to_25(x_test)
    Debugger.info_print('usr_dataset four return shape \n{}\n{}\n{}\n{}'.format( 
        x_train.shape, y_train_return.shape, x_test.shape, y_test_return.shape))

    return x_train, y_train_return, x_test, y_test_return, z_train, z_test


if __name__ == "__main__":
    # for i in range(110):
    #     if i in MISS_HOUSE:
    #         continue
    #     x_train, y_train_return, x_test, y_test_return = load_house_dataset_by_name(str(i).zfill(3),length=1440)
    x_train, y_train, x_test, y_test_return, z_train, z_test=load_house_dataset_by_houses_ex(TEST_HOUSE=testhouse,TRAIN_HOUSE=trainhouse,assign_behavior='meal')

# 將x_train轉換成2維
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    positive_ratio = 0.2
    n_negative = int(len(y_train[y_train==1]) / positive_ratio - len(y_train[y_train==1]))
    rus = RandomUnderSampler(sampling_strategy={0: n_negative, 1: len(y_train[y_train==1])}, random_state=42)
    x_train_res_flattened, y_train_res = rus.fit_resample(x_train_flattened, y_train)
    x_train_res = x_train_res_flattened.reshape(-1, x_train.shape[1], x_train.shape[2])

    print(f"Training: {np.mean(y_train_res)} positive ratio with {len(y_train_res)}")
    print(x_train_res.shape)
    print(y_train_res.shape)
