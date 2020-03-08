# @Last Modified Time : 2020/3/8 17:15
# @Author : Hongzuo Xu
# @Description ：
# This is the code for algorithm MIX which is an anomaly detection method for mixed-type data.
# This is the source code of the paper named "MIX: A joint learning framework for detecting outliers in both
# clustered and scattered outliers in mixed-type data" is published in ICDM19.
# Please cite our paper if you find this code is useful.
# ---- how to run this script?---
# You may want to find the sample input data set format in "data" folder.
# The name of categorical attributes name should be named as "A1", "A2", ..., and the numerical ones are "B1", "B2", ...
# The input path can be an individual data set or just a folder.
# The performance might have slight difference between two independent runs,
# in our paper, we report the average auc with std over 10 runs.
# ---- any parameters to tune?---
# k and epsilon are parameters, k = 0.3 and epsilon=0.01 is recommended,
# but for large data sets, please consider increase epsilon, say 0.05

from DataLoader import DataLoader
import MIX

import os
import time
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score

RUN_TIMES = 1
batch_size = 64
episode_max = 10000
epsilon = 0.01
k = 0.3


def main(input_path):
    t0 = time.time()
    data = DataLoader(input_path)
    data.data_prepare()
    t1 = time.time()

    auc_list = np.zeros(RUN_TIMES)
    time_list = np.zeros(RUN_TIMES)
    for i in range(RUN_TIMES):
        time_0 = time.time()
        score, n_iter = MIX.fit(data, batch_size=batch_size, episode_max=episode_max,
                                                    epsilon=epsilon, k=k, verbose=False)

        # perform MIX'
        # score, n_iter = OutlierEstimator.fit_prime(data, batch_size=64, episode_max=10000,
        #                                                  k=k, verbose=False)

        time_1 = time.time()
        auc_list[i] = roc_auc_score(data.list_of_class, score)
        time_list[i] = (t1 - t0) + (time_1 - time_0)

    print_text = "{}, {:.4},{:.4}, {:.4}s".format(data.data_name,
          np.average(auc_list), np.std(auc_list), np.average(time_list))

    print(print_text)

    doc = open('out.txt', 'a')
    print(print_text, file=doc)
    doc.close()
    return


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    input_root = "data/heart.csv"
    print("Parameters: episilon:{}, k:{}".format(epsilon, k))
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if os.path.isdir(input_root):
        for file_name in os.listdir(input_root):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                main(input_path)
    else:
        input_path = input_root
        main(input_path)
