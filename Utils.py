# @Time : 2019/4/13 14:25
# @Author : Hongzuo Xu
# @Description ：

import numpy as np
import pandas as pd
from scipy.io import loadmat
from collections import Counter
from sklearn import preprocessing
import numba as nb
import time
import math

class Data:
    def __init__(self, X):
        self.batch_start_index = 0
        self.X = X
        self.data_size = len(X)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        end_index = self.batch_start_index + batch_size
        if end_index > self.data_size:
            # Shuffle the data
            perm = np.arange(self.data_size)
            np.random.shuffle(perm)
            self.X = self.X[perm]

            self.epochs_completed += 1
            self.batch_start_index = 0

        start_index = self.batch_start_index
        batch_X = self.X[start_index: end_index]

        return batch_X


def mat2csv(in_path, out_path):
    df = loadmat(in_path)
    data = np.array(df['X'])
    label = np.array(df['Y'])
    label = label.reshape([8671,1])
    all = np.hstack((data, label))
    df = pd.DataFrame(all)

    df.to_csv(out_path, index=False)
    return


@nb.njit()
def normalise(vector, method="max_min"):
    if method == "max_min":
        [_max, _min] = [np.max(vector), np.min(vector)]
        vector = np.array([(vector[i] - _min) / (_max - _min) for i in range(len(vector))])
    elif method == "sum":
        sum = np.sum(vector)
        vector = np.array([item / sum for item in vector])
    else:
        raise ValueError("unsupported normalisation method.")
    return vector


def shuffle(data_matrix):
    perm = np.arange(len(data_matrix))
    np.random.shuffle(perm)
    data_matrix = data_matrix[perm]
    return data_matrix


def counter(in_path):
    data = pd.read_csv(in_path)
    label = data.values[: ,-1]
    print(Counter(label))


def kdd99_preprocess2(in_path):
    data = pd.read_csv(in_path)

    # label = data_matrix[: ,-1]
    # print(Counter(label))

    # order = ["A2","A3","A4","A7","A12","A21","A22","A1",'A5','A6', 'A8', 'A9', 'A10', 'A11',
    #          'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
    #          'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31',
    #          'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41','class']
    # data = data[order]
    # data.to_csv("data/org/kdd99_adjust.csv", index=False)

    new_df = data[data['class'].isin([0,4])]
    new_df.replace('4', '1', inplace=True)
    # new_df.replace(outlier_class, '1', inplace=True)
    out_df = new_df.copy()
    # new_df['nclass'] = new_df['class']
    # new_df.loc[new_df['class'] == 'normal.','nclass'] = 0
    # new_df.loc[new_df['class'] == 'satan.','nclass'] = 1
    # new_df = new_df.drop('class', axis=1)

    print(out_df)
    # drop columns with single value
    head = new_df.columns
    [n_o, n_f] = new_df.shape
    for i in range(n_f-1):
        values = new_df.values[:, i]
        if head[i].startswith("A"):
            if len(Counter(values)) == 1:
                print("Drop {}".format(head[i]))
                out_df = out_df.drop(head[i], axis=1)
        else:
            _max = np.max(values)
            _min = np.min(values)
            if _max == _min:
                print("Drop {}".format(head[i]))
                out_df = out_df.drop(head[i], axis=1)

    out_df.to_csv("data/covertype/ct_od" + ".csv", index=False)
    # out_df.to_csv("data/kdd99/kdd99_smtp" + ".csv", index=False)
    return


# def kdd99_preprocess(in_path):
#     data = pd.read_csv(in_path)
#
#     # label = data_matrix[: ,-1]
#     # print(Counter(label))
#
#     # order = ["A2","A3","A4","A7","A12","A21","A22","A1",'A5','A6', 'A8', 'A9', 'A10', 'A11',
#     #          'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
#     #          'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31',
#     #          'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41','class']
#     # data = data[order]
#     # data.to_csv("data/org/kdd99_adjust.csv", index=False)
#
#     outlier_class = "portsweep."
#     new_df = data[data['class'].isin(['normal.', outlier_class])]
#     new_df.replace('normal.', '0', inplace=True)
#     new_df.replace(outlier_class, '1', inplace=True)
#     out_df = new_df.copy()
#     # new_df['nclass'] = new_df['class']
#     # new_df.loc[new_df['class'] == 'normal.','nclass'] = 0
#     # new_df.loc[new_df['class'] == 'satan.','nclass'] = 1
#     # new_df = new_df.drop('class', axis=1)
#
#     # drop columns with single value
#     head = new_df.columns
#     [n_o, n_f] = new_df.shape
#     for i in range(n_f-1):
#         values = new_df.values[:, i]
#         if head[i].startswith("A"):
#             if len(Counter(values)) == 1:
#                 print("Drop {}".format(head[i]))
#                 out_df = out_df.drop(head[i], axis=1)
#         else:
#             _max = np.max(values)
#             _min = np.min(values)
#             if _max == _min:
#                 print("Drop {}".format(head[i]))
#                 out_df = out_df.drop(head[i], axis=1)
#
#     out_df.to_csv("data/kdd99/kdd99_" + outlier_class + ".csv", index=False)
#     return


# def preprocess(in_path, out_path):
#     data = pd.read_csv(in_path)
#     data.replace('nonad.', '0', inplace=True)
#     data.replace('ad.', '1', inplace=True)
#     label = data.values[:,-1]
#     data = data.drop("class", axis=1)
#     data.insert(0,"class", label)
#     data.to_csv(out_path, index=False)
#     return


def onehot_mixed2nume(in_path, out_path):
    data_name = in_path.split("/")[-1].split(".")[0]
    data = pd.read_csv(in_path)
    head = data.columns
    cf = [f for f in head if f.startswith("A")]
    for f in cf:
        le = preprocessing.LabelEncoder().fit_transform(data[f])
        if len(Counter(le)) > 2:
            oh = preprocessing.OneHotEncoder(sparse=False).fit_transform(le.reshape((-1,1)))
            v_num = oh.shape[1]
            data = data.drop(f, axis=1)
            for i in range(v_num):
                data.insert(0, f + "_" +str(i), oh[:, i])
        elif len(Counter(le)) == 2:
            data = data.drop(f, axis=1)
            data.insert(0, f, le)
        else:
            raise ValueError("feature with single value")

    out_path = out_path + data_name + "-oh_nume.csv"
    data.to_csv(out_path, index=False)
    return


def downsample_data(in_path, out_path_root, rate, times=10):
    '''
    Function to downsample balanced data to imbalanced dataset with a given imbalance rate
    :param in_path:  str, path of input data, should be formated in csv with head and labels in the rightmost column
    :param out_path_root: str, path to store downsampled imbalanced data
    :param rate:  float, imbalanced rate of new dataset, i.e. outlier_num / inlier_num
    :param times: int, optional, times of downsampling
    :return: none
    e.g. downsample_data("data/org/cylinder_bands.csv", "data/", 0.1)
    '''
    data_name = in_path.split("/")[-1].split(".")[0]
    data = pd.read_csv(in_path)
    data_matrix = data.values
    obj_num = data.shape[0]

    label_map = Counter(data_matrix[:, -1])
    # outlier_label = min(label_map, key=label_map.get)
    outlier_label = 1
    new_out_num = int((obj_num - label_map.get(outlier_label)) * rate)
    out_index = [ii for ii, obj in enumerate(data_matrix) if obj[-1] == outlier_label]

    for i in range(times):
        np.random.shuffle(out_index)
        drop_index = out_index[new_out_num:]
        new_df = data.drop(drop_index)
        out_path = out_path_root + data_name + "-" + str(i) + ".csv"
        new_df.to_csv(out_path, index=False)

    return


def get_mixed_data(in_path, out_path_root):
    data_name = in_path.split("/")[-1].split(".")[0]
    data = pd.read_csv(in_path)
    head = data.columns
    new_df = pd.DataFrame()
    cate_count = 0
    nume_count = 0

    for f in head:
        values = np.array(data[f])
        count_map = Counter(values)
        if f == "id":
            print('id')
            continue
        if len(count_map) == 1:
            continue

        if f == " Label":
            print("Label")
            print(count_map)
            new_df.insert(cate_count + nume_count, "class", values)
            continue

        if len(count_map) <= 2:
            print("cate", "A"+str(cate_count), 0)
            new_df.insert(0, "A"+str(cate_count), values)
            cate_count += 1
        else:
            print(f, "nume", "B" + str(nume_count), cate_count)
            new_df.insert(cate_count, "B"+str(nume_count), values)
            nume_count += 1

    # new_df.replace('BENIGN', '0', inplace=True)
    # new_df.replace('Web Attack', '1', inplace=True)

    out_path = out_path_root + data_name + "-mixed.csv"
    new_df.to_csv(out_path, index=False)


def discretise(in_path, out_path_root):
    data_name = in_path.split("/")[-1].split(".")[0]
    data = pd.read_csv(in_path)
    head = data.columns
    for f in head:
        if f.startswith("B"):
            values = np.array(data[f])
            _max = np.max(values)
            _min = np.min(values)
            print(f, _max, _min)

            values = (values - _min) / (_max - _min)
            avg = np.average(values)
            std = np.average(values)
            new_values = np.zeros([values.shape[0]], dtype=int)
            for ii, value in enumerate(values):
                # if avg - std <= value <= avg + std:
                #     new_values[ii] = 0
                # elif avg + std < value <= avg + 2 * std and avg - 2*std <= value < avg - std:
                #     new_values[ii] = 1
                # elif avg + 2*std < value <= avg + 3 * std and avg - 3*std <= value < avg - 2*std:
                #     new_values[ii] = 2
                # else:
                #     new_values[ii] = 3
                if avg - 3*std <= value <= avg + 3*std:
                    new_values[ii] = 0
                else:
                    new_values[ii] = 1
            data = data.drop(f, axis=1)
            if np.max(new_values) != np.min(new_values):
                data.insert(0, f, new_values)
    out_path = out_path_root + data_name + "-dc" + ".csv"
    data.to_csv(out_path, index=False)

    return


def get_sorted_index(score, order='descending'):
    '''
    :param score:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index':i, 'score':score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


# @nb.njit()
def get_rank(score):
    '''
    :param score:
    :return:
    e.g. input: [0.8, 0.4, 0.6] return [0, 2, 1]
    '''
    sort = np.argsort(score)
    size = score.shape[0]
    rank = np.zeros(size)
    for i in range(size):
        rank[sort[i]] = size - i - 1

    return rank


def get_five_number_summary(results):
    results.sort()
    results = np.array(results)
    median = np.median(results)
    half_size = math.floor(0.5 * len(results))
    interquartile1 = np.median(results[0:half_size])
    interquartile2 = np.median(results[half_size:len(results)])

    iqr = interquartile2 - interquartile1
    results = results.tolist()
    removed = []
    while True:
        max = np.max(results)
        if max > interquartile2 + 1.5 * iqr:
            removed.append(max)
            results.remove(max)
        else:
            break

    while True:
        min = np.min(results)
        if min < interquartile1 - 1.5 * iqr:
            removed.append(min)
            results.remove(min)
        else:
            break

    return median, max, min, interquartile1, interquartile2, removed


if __name__ == "__main__":
    # mat2csv("E:/data/categorical data-survey/caltech101_silhouettes_28.mat", "data/cal28-full.csv")
    # get_mixed_data("E:/data/categorical data-survey/Arrhythmia-od - less_than20&others.csv", "E:/data\categorical data-survey/new.csv")
    # downsample_data("data/org/covtype-nm_adjust.csv", "data/", 0.1)
    # kdd99_preprocess2("E:/1-anomaly detection/08-CIKM19/data/mixed data/covertype/covtype-nm-maxmin-od.csv")
    counter("E:/data/categorical data-survey/cal28-full.csv")
    # get_mixed_data("data/test/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "data/IDS2017/")


