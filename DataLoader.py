# @Time : 2019/3/19 21:15
# @Author : Hongzuo Xu
# @Description ：Toolkit to load mixed-type data

import pandas as pd
from collections import Counter
import numpy as np


class DataLoader:

    # import data, get labels and count number of features, values, and objects
    def __init__(self, data_path, verbose=False):
        input_split = data_path.split("/")
        self.data_name = input_split[len(input_split) - 1].split(".")[0]

        data = pd.read_csv(data_path)
        self.data_matrix = data.values
        [self.objects_num, self.all_features_num] = data.shape
        self.all_features_num = self.all_features_num - 1

        self.head = data.columns
        self.cate_features_num = len([f_name for f_name in self.head if f_name.startswith("A")])
        self.nume_features_num = len([f_name for f_name in self.head if f_name.startswith("B")])

        self.values_num_list = np.array([len(np.unique(self.data_matrix[:, i])) for i in range(self.cate_features_num)])
        self.values_num = np.sum(self.values_num_list)
        self.list_of_class = np.array(self.data_matrix[:, self.all_features_num], dtype='int')

        self.cate_data = np.zeros([self.objects_num, self.cate_features_num], dtype=int)
        self.nume_data = np.zeros([self.objects_num, self.nume_features_num])

        self.value_frequency_list = np.zeros(self.values_num)
        self.value_list = []
        self.first_value_index = []
        self.first_value_index.append(0)
        self.value_feature_indicator = []

        self.verbose = verbose
        return

    # calculate basic statistical information
    def data_prepare(self):
        # calc first_value_index, count value frequency,
        # generate value list for each feature, indicate the feature index of the values
        for i in range(self.cate_features_num):
            column = self.data_matrix[:, i]
            this_value_list = np.unique(column).tolist()
            feature_value_num = len(this_value_list)
            self.first_value_index.append(self.first_value_index[i] + feature_value_num)
            for j in range(feature_value_num):
                self.value_feature_indicator.append(i)

            frequency_map = Counter(column)
            for jj, item in enumerate(this_value_list):
                frequency = frequency_map.get(item)
                self.value_frequency_list[self.first_value_index[i] + jj] = frequency
            self.value_list.append(this_value_list)

        # process categorical space
        for i in range(0, self.cate_features_num):
            this_value_list = self.value_list[i]
            this_value_index_map = {}
            for j in range(len(this_value_list)):
                this_value_index_map[this_value_list[j]] = self.first_value_index[i] + j
            for k in range(self.objects_num):
                self.cate_data[k][i] = this_value_index_map[self.data_matrix[k][i]]

        # normalise numerical features using max-min normalisation method, normalised features will range from 0 to 1
        for i in range(self.cate_features_num, self.all_features_num):
            column_max = np.max(self.data_matrix[:, i])
            column_min = np.min(self.data_matrix[:, i])
            if column_max - column_min == 0:
                raise ValueError("all values in feature {} ({}) are zero.".format(i, self.head[i]))
            for j in range(self.objects_num):
                self.nume_data[j][i-self.cate_features_num] = \
                    float(self.data_matrix[j][i] - column_min) / float(column_max - column_min)

        return
