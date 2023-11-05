import argparse
import numpy as np

from .. import DATASET_INFO


def concept2question_from_Q(Q_table):
    # 将Q table转换为{concept_id1: [question_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[:, i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[1])}
    return result


def question2concept_from_Q(Q_table):
    # 将Q table转换为{question_id1: [concept_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[0])}
    return result


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
