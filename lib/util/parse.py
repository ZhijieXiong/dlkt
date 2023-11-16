import argparse
import numpy as np

from collections import defaultdict

from .. import DATASET_INFO
from . import data as data_util


def concept2question_from_Q(Q_table):
    # 将Q table转换为{concept_id1: [question_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[:, i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[1])}
    return result


def question2concept_from_Q(Q_table):
    # 将Q table转换为{question_id1: [concept_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[0])}
    return result


def get_concept_from_question(question_id, Q_table):
    return np.argwhere(Q_table[question_id] == 1).reshape(-1).tolist()


def get_question_from_concept(concept_id, Q_table):
    return np.argwhere(Q_table[:, concept_id] == 1).reshape(-1).tolist()


def get_keys_from_uniform(data_uniformed):
    item_data = data_uniformed[0]
    id_keys = []
    for k in item_data.keys():
        if type(item_data[k]) is not list:
            id_keys.append(k)
    seq_keys = list(set(item_data.keys()) - set(id_keys))
    return id_keys, seq_keys


def parse_data_type(dataset_name, data_type):
    """
    判断一个数据集是否有某种数据类型（multi concept、single concept、only question）
    :param dataset_name:
    :param data_type:
    :return:
    """
    datasets_has_concept = DATASET_INFO.datasets_has_concept()
    datasets_multi_concept = DATASET_INFO.datasets_multi_concept()

    if data_type == "multi_concept":
        return dataset_name in datasets_multi_concept
    elif data_type == "single_concept":
        return dataset_name in datasets_has_concept
    elif data_type == "only_question":
        return (dataset_name in datasets_multi_concept) or (dataset_name not in datasets_has_concept)
    else:
        assert False, f"data type \"{data_type}\" does not exist!"


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cal_accuracy4data(data_uniformed, min_seq_len=10):
    accuracy_list = []
    count_statics = 0
    for item_data in data_uniformed:
        seq_len = item_data["seq_len"]
        if seq_len < min_seq_len:
            continue
        num_right = 0
        num_wrong = 0
        for i, m in enumerate(item_data["mask_seq"]):
            if m == 0:
                break
            num_right += item_data["correct_seq"][i]
            num_wrong += (1 - item_data["correct_seq"][i])
        accuracy = num_right / (num_right + num_wrong)
        item_data["acc"] = accuracy
        accuracy_list.append(accuracy)
        count_statics += 1
    return accuracy_list


def cal_distinction(data_uniformed, data_type, min_count2drop=30, min_seq_len=10):
    # 公式：总分最高的27%学生（H）和总分最低的27%学生（L），计算H和L对某道题的通过率，之差为区分度，区分度大于0.4为高区分度习题，低于0.2表示区分度不好
    concepts_high_distinction = []
    questions_high_distinction = []

    def cal_diff(D, k):
        # 计算正确率，习题或者知识点
        seqs = [item[k] for item in D]
        correct_seqs = [item["correct_seq"] for item in D]
        corrects = defaultdict(int)
        counts = defaultdict(int)

        for seq, correct_seq in zip(seqs, correct_seqs):
            for k_id, correct in zip(seq, correct_seq):
                corrects[k_id] += correct
                counts[k_id] += 1

        # 丢弃练习次数少于min_count次的习题或者知识点
        all_ids = list(counts.keys())
        for k_id in all_ids:
            if counts[k_id] < min_count2drop:
                del counts[k_id]
                del corrects[k_id]

        return {k_id: corrects[k_id] / float(counts[k_id]) for k_id in corrects}

    def get_high_distinction(H, L, update_target):
        intersection_H_L = set(H.keys()).intersection(set(L.keys()))
        for k_id in intersection_H_L:
            if H[k_id] - L[k_id] >= 0.35:
                update_target.append(k_id)

    def get_high_low_accuracy_seqs(acc_list, data_added_acc):
        acc_list = sorted(acc_list)
        count_statics = len(acc_list)
        high_acc = acc_list[int(count_statics * (1 - 0.27))]
        low_acc = acc_list[int(count_statics * 0.27)]
        H_acc = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] >= high_acc, data_added_acc))
        L_acc = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] <= low_acc, data_added_acc))
        return H_acc, L_acc

    dataset_concept = data_util.dataset_delete_pad(data_uniformed)
    # 统计知识点正确率
    accuracy_list = cal_accuracy4data(dataset_concept, min_seq_len)
    H_concept, L_concept = get_high_low_accuracy_seqs(accuracy_list, dataset_concept)
    H_concept_diff = cal_diff(H_concept, "concept_seq")
    L_concept_diff = cal_diff(L_concept, "concept_seq")
    get_high_distinction(H_concept_diff, L_concept_diff, concepts_high_distinction)

    # 如果是多知识点数据集，并且数据格式是multi concept，需要获取习题序列
    if data_type == "multi_concept":
        dataset_question = data_util.dataset_delete_pad(data_util.dataset_agg_concept(data_uniformed))
    else:
        dataset_question = dataset_concept

    # 统计习题正确率
    if data_type == "multi_concept":
        accuracy_list = cal_accuracy4data(dataset_question, min_seq_len)
        H_question, L_question = get_high_low_accuracy_seqs(accuracy_list, dataset_question)
    else:
        H_question = H_concept
        L_question = L_concept
    H_question_diff = cal_diff(H_question, "question_seq")
    L_question_diff = cal_diff(L_question, "question_seq")
    get_high_distinction(H_question_diff, L_question_diff, questions_high_distinction)

    return concepts_high_distinction, questions_high_distinction
