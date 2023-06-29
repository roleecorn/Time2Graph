import pandas as pd
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
MISS_HOUSE=[0,49, 48, 40, 50, 41, 44, 46, 43, 4, 42, 47, 45,75]
TRAIN_HOUSE=[62, 84, 82, 17, 52, 35, 11, 94, 13, 79, 109, 32, 30, 101, 7, 58, 83, 37, 18,
             36, 9, 68, 29, 21, 93, 27, 66, 15, 26, 5, 19, 51, 10, 90, 8, 92, 38, 99, 69,
             67, 56, 63, 80, 59, 81, 74, 108, 91, 54, 71, 61, 98, 70, 34, 105, 72, 77, 104,
             2, 65, 23, 3, 25, 102, 88, 100, 96, 22, 73, 85, 16, 106, 64, 110,31, 86,39,]
TEST_HOUSE=[76, 53, 107, 20, 95, 97, 78, 55, 12, 1, 33, 89, 60, 28, 57, 6, 24, 14, 103, 87]
def assign_meal(row):
    if row['home(meal)'] == 1:
        return 1
    else :
        return 0
def assign_other(row):
    if row['home(other)'] == 1:
        return 1
    else :
        return 0
def assign_out(row):
    if row['out'] == 1:
        return 1
    else :
        return 0
def assign_sleep(row):
    if row['home(sleep)'] == 1:
        return 1
    else :
        return 0
assign={'sleep':assign_sleep,'out':assign_out,'meal':assign_meal,'other':assign_other}

# def load_house_dataset_by_houses(TRAIN_HOUSE, TEST_HOUSE, assign_behavior):
#     """
#     load TEPCO dataset given household number.
#     :param TRAIN_HOUSE:
#         list of training household ids
#     :param TEST_HOUSE:
#         list of testing household ids
#     :param length:
#         time series length that want to load in.
#     :return:
#         x_train, y_train_return, x_test, y_test_return
#     """

#     dir_path = '{}/TEPCO_data'.format(module_path)

#     x_train, y_train, x_test, y_test = [], [], [], []

#     # Load the training data
#     mapping =assign[assign_behavior]
#     for fname in TRAIN_HOUSE:
#         labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
#                              usecols=['home(sleep)','home(other)','home(meal)','out'])
#         labels['behavior'] = labels.apply(mapping, axis=1)
#         datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
#         x = datas.values.reshape(-1, 15, 1).astype(np.float)
#         y = labels['behavior'].values.astype(np.int)
#         x_train.append(x)
#         y_train.append(y)

#     # Load the testing data
#     for fname in TEST_HOUSE:
#         labels = pd.read_csv('{}/behavior/{}/001.csv'.format(dir_path, int(fname)),
#                              usecols=['home(sleep)','home(other)','home(meal)','out'])
#         labels['behavior'] = labels.apply(mapping, axis=1)
#         datas = pd.read_csv('{}/1day/{}.csv'.format(dir_path, fname))['total_power']
#         x = datas.values.reshape(-1, 15, 1).astype(np.float)
#         y = labels['behavior'].values.astype(np.int)
#         x_test.append(x)
#         y_test.append(y)

#     # Convert list of arrays to one large array
#     x_train = np.concatenate(x_train, axis=0)
#     y_train = np.concatenate(y_train, axis=0)
#     x_test = np.concatenate(x_test, axis=0)
#     y_test = np.concatenate(y_test, axis=0)

#     # Convert labels to start from 0
#     lbs = np.unique(np.concatenate([y_train, y_test], axis=0))
#     y_train_return, y_test_return = np.copy(y_train), np.copy(y_test)
#     for idx, val in enumerate(lbs):
#         y_train_return[y_train == val] = idx
#         y_test_return[y_test == val] = idx

#     Debugger.info_print('usr_dataset four return shape \n{}\n{}\n{}\n{}'.format( 
#         x_train.shape,y_train_return.shape,x_test.shape,y_test_return.shape))
        
#     return x_train, y_train_return, x_test, y_test_return
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
        x = datas.values.reshape(-1, 15, 1).astype(np.float)
        y = labels['behavior'].values.astype(np.int)
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
        x = datas.values.reshape(-1, 15, 1).astype(np.float)
        y = labels['behavior'].values.astype(np.int)
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


if __name__ == "__main__":
    # for i in range(110):
    #     if i in MISS_HOUSE:
    #         continue
    #     x_train, y_train_return, x_test, y_test_return = load_house_dataset_by_name(str(i).zfill(3),length=1440)
    x_train, y_train_return, x_test, y_test_return = load_house_dataset_by_name(str(1).zfill(3),length=1440)
    print(y_test_return)
    load_house_dataset_by_houses(TEST_HOUSE=TEST_HOUSE,TRAIN_HOUSE=TRAIN_HOUSE)