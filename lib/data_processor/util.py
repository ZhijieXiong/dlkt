import pandas as pd
from copy import deepcopy
from collections import defaultdict

from ..util.parse import get_keys_from_uniform


def get_info_function(df, col_name):
    return len(list(filter(lambda x: str(x) != "nan", pd.unique(df[col_name]))))


def process4DIMKT(data_uniformed, num_q_level=100, num_c_level=100):
    data_uniformed = deepcopy(data_uniformed)
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    has_concept = "concept_seq" in seq_keys
    questions_frequency = defaultdict(int)
    questions_correct = defaultdict(int)
    concepts_frequency = defaultdict(int)
    concepts_correct = defaultdict(int)
    for item_data in data_uniformed:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            correct = item_data["correct_seq"][i]
            questions_frequency[q_id] += 1
            questions_correct[q_id] += correct
            if has_concept:
                c_id = item_data["concept_seq"][i]
                concepts_frequency[c_id] += 1
                concepts_correct[c_id] += correct

    for q_id in questions_frequency.keys():
        questions_correct[q_id] = int((num_q_level - 1) * questions_correct[q_id] / questions_frequency[q_id])
    if has_concept:
        for c_id in concepts_frequency.keys():
            concepts_correct[c_id] = int((num_c_level - 1) * concepts_correct[c_id] / concepts_frequency[c_id])

    data_processed = []
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        item_data_new["question_diff_seq"] = []
        if has_concept:
            item_data_new["concept_diff_seq"] = []

        for i in range(item_data["seq_len"]):
            for k in seq_keys:
                if k == "question_seq":
                    q_id = item_data["question_seq"][i]
                    item_data_new["question_diff_seq"].append(questions_correct[q_id])
                if k == "concept_seq":
                    c_id = item_data["concept_seq"][i]
                    item_data_new["concept_diff_seq"].append(concepts_correct[c_id])
                item_data_new[k].append(item_data[k][i])
        data_processed.append(item_data_new)

    return data_processed


def id_remap4data_uniformed(data_uniformed, seq_names2remap):
    """
    将指定的seq id进行重映射，如LPKT用到的time相关信息，有很多time id在数据集中没出现，就可以重映射，减少id数量，从而减少后面训练模型用的显存
    以及将multi_concept处理成single_concept，也就是concept（为"1_2"这种形式）重映射
    :param data_uniformed:
    :param seq_names2remap:
    :return:
    """
    ids_all = {seq_name: [] for seq_name in seq_names2remap}
    for item_data in data_uniformed:
        for i in range(item_data["seq_len"]):
            for seq_name in seq_names2remap:
                seq_id = item_data[seq_name][i]
                ids_all[seq_name].append(seq_id)

    id_remap_all = {
        seq_name: {
            id_original: id_mapped for id_mapped, id_original in enumerate(list(set(ids_all[seq_name])))
        }
        for seq_name in seq_names2remap
    }

    data_new = []
    for item_data in data_uniformed:
        item_data_new = {}
        for k in item_data.keys():
            if k not in seq_names2remap:
                item_data_new[k] = deepcopy(item_data[k])
            else:
                item_data_new[k] = list(map(lambda x: id_remap_all[k][x], item_data[k]))
        data_new.append(item_data_new)

    return data_new, id_remap_all
