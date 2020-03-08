import time
import numpy as np
import numba as nb
import Utils
from Utils import Data
from sklearn.metrics import roc_auc_score
from DeepAutoEncoder import DeepAutoEncoder


def fit(data, batch_size, episode_max, epsilon, k, verbose=False):
    s_time = time.time()
    if verbose:
        print("-------------{}--------------".format(data.data_name))
        print("f:{}, c_f:{}, n_f:{}, o:{}".format(
            data.all_features_num, data.cate_features_num, data.nume_features_num, data.objects_num))

    nume_data = data.nume_data
    if data.cate_features_num > 50:
        init_cate_scores = init_cate_scoring_jit(data.cate_data, data.value_frequency_list)
    else:
        init_cate_scores = init_cate_scoring(data.cate_data, data.value_frequency_list)
    init_nume_scores = init_nume_od(data.nume_data)
    last_cate_scores = init_cate_scores
    init_scores = ensemble_scores(init_cate_scores, init_nume_scores)

    index_list = get_normal_data(init_scores, k)
    normal_set = set(index_list)
    normal_data = nume_data[index_list]

    if verbose:
        print("#init_cate: {:.4}, init_nume:{:.4}, init:{:.4}".format(
            roc_auc_score(data.list_of_class, init_cate_scores),
            roc_auc_score(data.list_of_class, init_nume_scores),
            roc_auc_score(data.list_of_class, init_scores)))
    count = 0
    deep_ae = DeepAutoEncoder(input_dimension=data.nume_features_num)

    while True:
        time1 = time.time()
        ae_scores = ae_od(normal_data=normal_data, all_data=nume_data, ae_model=deep_ae,
                          EPISODE_MAX=episode_max, BATCH_SIZE=batch_size, verbose=False)

        e_s = ensemble_scores(ae_scores, last_cate_scores)
        tt1 = time.time()
        cate_scores = cate_od(data.cate_data, data.value_frequency_list, e_s, k)
        tt2 = time.time()

        out = ensemble_scores(cate_scores, ae_scores)
        index_list = get_normal_data(out, k)
        tt3 = time.time()
        new_diff = list(set(index_list) - (normal_set & set(index_list)))
        normal_set = normal_set | set(index_list)
        time2 = time.time()

        if verbose:
            print("IterRESULTS: #{}, out:{:.4}, ae:{:.4}, e_s:{:.4}, cate:{:.4}, normal_set:{}, newdiff:{}, {:.4}s".format(
                count,
                roc_auc_score(data.list_of_class, out),
                roc_auc_score(data.list_of_class, ae_scores),
                roc_auc_score(data.list_of_class, e_s),
                roc_auc_score(data.list_of_class, cate_scores),
                len(normal_set), len(new_diff), time2 - time1))
            print("@@TIME: ae:{:.4}s, cate:{:.4}s, {:.4}s, {:.4}s".format(tt1-time1, tt2-tt1, tt3-tt2, time2-tt3))

        if len(new_diff) < epsilon*data.objects_num:
            break

        normal_data = nume_data[list(normal_set)]
        last_cate_scores = cate_scores
        count += 1

    e_time = time.time()
    if verbose:
        auc = roc_auc_score(data.list_of_class, out)
        print("{},{},{},{},{},{:.4},{:.4}s".format(data.data_name, data.all_features_num, data.cate_features_num,
                                                   data.nume_features_num, data.objects_num,
                                                   auc, (e_time - s_time)))

    return out, count


def fit_prime(data, batch_size=64, episode_max=10000, k=0.3, verbose=False):
    s_time = time.time()
    if verbose:
        print("-------------{}--------------".format(data.data_name))
        print("f:{}, c_f:{}, n_f:{}, o:{}".format(data.all_features_num, data.cate_features_num, data.nume_features_num, data.objects_num))

    nume_data = data.nume_data
    cate_data = data.cate_data
    if data.cate_features_num > 50:
        init_cate_scores = init_cate_scoring_jit(cate_data, data.value_frequency_list)
    else:
        init_cate_scores = init_cate_scoring(cate_data, data.value_frequency_list)
    init_nume_scores = init_nume_od(data.nume_data)
    init_scores = ensemble_scores(init_cate_scores, init_nume_scores)

    index_list = get_normal_data(init_scores, k)
    normal_data = nume_data[index_list]

    if verbose:
        print("#init_cate: {:.4}, init_nume:{:.4}, init:{:.4}".format(
            roc_auc_score(data.list_of_class, init_cate_scores),
            roc_auc_score(data.list_of_class, init_nume_scores),
            roc_auc_score(data.list_of_class, init_scores)))
    deep_ae = DeepAutoEncoder(input_dimension=data.nume_features_num)
    ae_scores = ae_od(normal_data=normal_data, all_data=nume_data, ae_model=deep_ae,
                      EPISODE_MAX=episode_max, BATCH_SIZE=batch_size, verbose=False)

    e_s = ensemble_scores(ae_scores, init_cate_scores)
    cate_scores = cate_od(cate_data, data.value_frequency_list, e_s, k)
    out = ensemble_scores(cate_scores, ae_scores)
    e_time = time.time()
    if verbose:
        auc = roc_auc_score(data.list_of_class, out)
        print("{},{},{},{},{},{:.4},{:.4}s".format(data.data_name, data.all_features_num, data.cate_features_num,
                                      data.nume_features_num, data.objects_num,
                                      auc, (e_time - s_time)))

    n_iter = 1
    return out, n_iter


@nb.njit()
def init_cate_scoring_jit(cate_data, value_frequency_list):
    scores = np.zeros(len(cate_data))
    shape = cate_data.shape
    for ii in range(shape[0]):
        obj = cate_data[ii]
        obj_score = 0.
        for f in range(shape[1]):
            obj_score += 1. / value_frequency_list[obj[f]]
        scores[ii] = obj_score / shape[1]
    return scores


def init_cate_scoring(cate_data, value_frequency_list):
    scores = np.zeros(len(cate_data))
    shape = cate_data.shape
    for ii in range(shape[0]):
        obj = cate_data[ii]
        obj_score = 0.
        for f in range(shape[1]):
            obj_score += 1. / value_frequency_list[obj[f]]
        scores[ii] = obj_score / shape[1]
    return scores


def init_nume_od(nume_data):
    mean = np.average(nume_data, axis=0)
    std = np.std(nume_data, axis=0)
    scores = np.zeros(nume_data.shape[0])
    for jj, obj in enumerate(nume_data):
        scores[jj] = np.average(abs(obj - mean) / std)
    return scores


def get_normal_data(scores, rate):
    all_size = scores.shape[0]
    sorted_index = Utils.get_sorted_index(scores, order="descending")
    size = int(all_size * rate)
    index_list = sorted_index[all_size - size: all_size]
    return index_list


def ae_od(normal_data, all_data, ae_model, EPISODE_MAX=10000, BATCH_SIZE=64, verbose=False):
    [n_obj, n_f] = all_data.shape
    train_data = Data(normal_data)
    loss_list = np.zeros(200)
    for episode in range(EPISODE_MAX):
        batch_x = train_data.next_batch(BATCH_SIZE)
        train_loss = ae_model.train_model(batch_x)
        loss_list[episode % 200] = train_loss
        avg = 0.
        std = 0.

        if episode % 200 == 0 and episode // 200 != 0:
            std = np.std(loss_list)
            avg = np.average(loss_list)
            if std < 0.05*avg or avg < 1e-5:
                if verbose:
                    print('  DeepAE:{}, episode: {}, loss: {:.4}, avg,std: {:.4}, {:.4}'.
                          format(batch_x.shape, episode, train_loss, avg, std))
                break
            loss_list = np.zeros(200)

        if episode % 2000 == 0 and verbose:
            print('  DeepAE:{}, episode: {}, loss: {:.4}, avg,std: {:.4}, {:.4}'.
                  format(batch_x.shape, episode, train_loss, avg, std))

    anomaly_scores = np.zeros([n_obj])
    for i, obj in enumerate(all_data):
        anomaly_scores[i] = ae_model.test_model(obj.reshape([1, n_f]))
    return anomaly_scores


def cate_od(cate_data, value_frequency_list, combine_score, k):
    threshold = np.median(combine_score)
    values_num = value_frequency_list.shape[0]
    [obj_num, cate_f_num] = cate_data.shape

    c_p = np.zeros(values_num)
    value_score = np.zeros(values_num)
    for jj, obj in enumerate(cate_data):
        obj_score = combine_score[jj]
        for f in range(cate_f_num):
            value_score[obj[f]] += obj_score
            if obj_score <= threshold:
                c_p[obj[f]] += 1

    c_p = c_p / value_frequency_list
    c_p = 1. - c_p

    value_score = value_score / value_frequency_list
    [_max, _min] = [np.max(value_score), np.min(value_score)]
    value_score = np.array([(value_score[i] - _min) / (_max - _min) for i in range(values_num)])

    value_final_score = value_score * c_p

    obj_score = np.zeros(obj_num)
    for ii, obj in enumerate(cate_data):
        tmp_s = 0.
        for f in range(cate_f_num):
            tmp_s += value_final_score[obj[f]]
        obj_score[ii] = tmp_s / cate_f_num

    return obj_score


def ensemble_scores(score1, score2):
    '''
    :param score1:
    :param score2:
    :return: ensemble score
    @@ ensemble two score functions
    we use a non-parameter way to dynamically get the tradeoff between two estimated scores.
    It is much more important if one score function evaluate a object with high outlier socre,
    which should be paid more attention on these scoring results.
    instead of using simple average, median or other statistics
    '''

    objects_num = len(score1)

    [_max, _min] = [np.max(score1), np.min(score1)]
    score1 = (score1 - _min) / (_max - _min)
    [_max, _min] = [np.max(score2), np.min(score2)]
    score2 = (score2 - _min) / (_max - _min)

    # sort1 = np.argsort(score1)
    # rank1 = np.zeros(objects_num)
    # for i in range(objects_num):
    #     rank1[sort1[i]] = objects_num - i - 1

    # sort2 = np.argsort(score2)
    # rank2 = np.zeros(objects_num)
    # for i in range(objects_num):
    #     rank2[sort2[i]] = objects_num - i - 1

    rank1 = Utils.get_rank(score1)
    rank2 = Utils.get_rank(score2)

    alpha_list = (1. / (2 * (objects_num - 1))) * (rank2 - rank1) + 0.5
    combine_score = alpha_list * score1 + (1. - alpha_list) * score2
    # combine_score = 1. / (alpha_list * (1. / score1) + (1. - alpha_list) * (1. / score2))
    return combine_score

