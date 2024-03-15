import pandas as pd
from copy import deepcopy
from collections import defaultdict

from ..util.parse import get_keys_from_uniform


def get_info_function(df, col_name):
    return len(list(filter(lambda x: str(x) != "nan", pd.unique(df[col_name]))))


def process4DIMKT(data_uniformed, num_q_level=100, num_c_level=100):
    """
    准备DIMKT需要的数据，即知识点和习题的难度id
    :param data_uniformed:
    :param num_q_level:
    :param num_c_level:
    :return:
    """
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


def process4Akt_assist2009(data_uniformed, concept_id2name, concept_id_map):
    """
    AKT中对于Assist2009是丢弃没有concept name的数据（知识点数量由123变为110）
    :param data_uniformed:
    :param concept_id2name:
    :param concept_id_map:
    :return:
    """
    # w: with
    c_ids_original_w_name = set(pd.unique(concept_id2name["concept_id"]))
    c_ids_mapped_w_name = []
    c_ids_mapped_all = set(pd.unique(concept_id_map["concept_mapped_id"]))
    for c_id_ori in c_ids_original_w_name:
        c_id_mapped = concept_id_map[concept_id_map["concept_id"] == c_id_ori]["concept_mapped_id"].iloc[0]
        c_ids_mapped_w_name.append(c_id_mapped)
    concept2drop = c_ids_mapped_all - set(c_ids_mapped_w_name)

    data_uniformed = deepcopy(data_uniformed)
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)

    data_remain = []
    num_drop_interactions = 0
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        for i in range(item_data["seq_len"]):
            c_id = item_data["concept_seq"][i]
            if c_id in concept2drop:
                num_drop_interactions += 1
                continue
            for k in seq_keys:
                item_data_new[k].append(item_data[k][i])
        item_data_new["seq_len"] = len(item_data_new["question_seq"])
        data_remain.append(item_data_new)

    return data_remain


def process4CL4kt_assist2009(data_uniformed, concept_id2name, concept_id_map):
    """
    CL4KT中对于Assist2009是先丢弃没有concept name的数据（知识点数量由123变为110），然后再将多知识点组合看成新知识点
    :param data_uniformed:
    :param concept_id2name:
    :param concept_id_map:
    :return:
    """
    # w: with
    c_ids_original_w_name = set(pd.unique(concept_id2name["concept_id"]))
    concept2drop = []
    for _, row in concept_id_map.iterrows():
        multi_c_id_mapped = row["concept_mapped_id"]
        multi_c_id = row["concept_id"]
        multi_c_id = list(map(int, multi_c_id.split("_")))
        for c_id_ori in multi_c_id:
            if c_id_ori not in c_ids_original_w_name:
                concept2drop.append(multi_c_id_mapped)
                break

    data_uniformed = deepcopy(data_uniformed)
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)

    data_remain = []
    num_drop_interactions = 0
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        for i in range(item_data["seq_len"]):
            c_id = item_data["concept_seq"][i]
            if c_id in concept2drop:
                num_drop_interactions += 1
                continue
            for k in seq_keys:
                item_data_new[k].append(item_data[k][i])
        item_data_new["seq_len"] = len(item_data_new["question_seq"])
        data_remain.append(item_data_new)

    return data_remain


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
