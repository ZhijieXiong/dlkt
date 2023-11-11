import os
import json
import numpy
import numpy as np
from copy import deepcopy
from collections import defaultdict

from .parse import get_keys_from_uniform


def write2file(data, data_path):
    # id_keys表示序列级别的特征，如user_id, seq_len
    # seq_keys表示交互级别的特征，如question_id, concept_id
    id_keys = []
    seq_keys = []
    for key in data[0].keys():
        if type(data[0][key]) == list:
            seq_keys.append(key)
        else:
            id_keys.append(key)
    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + ";" + ",".join(seq_keys) + "\n"
        f.write(first_line)
        for item_data in data:
            for k in id_keys:
                f.write(f"{item_data[k]}\n")
            for k in seq_keys:
                f.write(",".join(map(str, item_data[k])) + "\n")


def read_preprocessed_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        seq_interaction_keys_str = first_line.split(";")
        id_keys_str = seq_interaction_keys_str[0].strip()
        seq_keys_str = seq_interaction_keys_str[1].strip()
        id_keys = id_keys_str.split(",")
        seq_keys = seq_keys_str.split(",")
        keys = id_keys + seq_keys
        num_key = len(keys)
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            if i % num_key == 0:
                item_data = {}
            line_content = list(map(int, line_str.strip().split(",")))
            if len(line_content) == 1:
                # 说明是序列级别的特征，即user id、seq len、segment index等等
                item_data[keys[int(i % num_key)]] = line_content[0]
            else:
                # 说明是interaction级别的特征，即question id等等
                item_data[keys[int(i % num_key)]] = line_content
            if i % num_key == (num_key - 1):
                data.append(item_data)

    return data


def load_json(json_path):
    with open(json_path, "r") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def get_concept_from_question(q_table: numpy.ndarray, question_id):
    return np.argwhere(q_table[question_id] == 1).reshape(-1).tolist()


def dataset_delete_pad(dataset):
    id_keys, seq_keys = get_keys_from_uniform(dataset)
    data_uniformed = []
    for item_data in dataset:
        item_data_new = deepcopy(item_data)
        mask_seq = item_data_new["mask_seq"]
        end_index = mask_seq.index(0) if mask_seq[-1] != 1 else len(mask_seq)
        for k in seq_keys:
            item_data_new[k] = item_data_new[k][0:end_index]
        item_data_new["seq_len"] = len(item_data_new["correct_seq"])
        data_uniformed.append(item_data_new)
    return data_uniformed


def data_pad(data_uniformed, max_seq_len=200, padding_value=0):
    dataset_new = []
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        item_new = {k: item_data[k] for k in id_keys}
        seq_len = len(item_data["correct_seq"])
        for k in seq_keys:
            item_new[k] = item_data[k] + [padding_value] * (max_seq_len - seq_len)
        dataset_new.append(item_new)
    return dataset_new


def dataset_agg_concept(data_uniformed):
    # 用于将数据中question_seq序列为-1的去掉，也就是生成single数据，不做multi
    data_new = []
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        item_data_new = {}
        for key in id_keys:
            item_data_new[key] = item_data[key]
        for key in seq_keys:
            item_data_new[key] = []
        for i, q_id in enumerate(item_data["question_seq"]):
            if q_id != -1:
                for key in seq_keys:
                    item_data_new[key].append(item_data[key][i])
        item_data_new["seq_len"] = len(item_data_new["correct_seq"])
        data_new.append(item_data_new)
    return data_new


def data_agg_question(data_uniformed):
    """
    将multi concept的数据中question seq里的-1替换为对应的q id
    :param data_uniformed:
    :return:
    """
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    if "question_seq" not in seq_keys:
        return data_uniformed

    data_converted = []
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            if k == "question_seq":
                question_seq = item_data["question_seq"]
                question_seq_new = []
                current_q = question_seq[0]
                for q in question_seq:
                    if q != -1:
                        current_q = q
                    question_seq_new.append(current_q)
                item_data_new["question_seq"] = question_seq_new
            else:
                item_data_new[k] = deepcopy(item_data[k])
        data_converted.append(item_data_new)

    return data_converted


def drop_qc(data_uniformed, num2drop=30):
    """
    丢弃练习次数少于指定值的习题，如DIMKT丢弃练习次数少于30次的习题
    :param data_uniformed:
    :param num2drop:
    :return:
    """
    data_uniformed = deepcopy(data_uniformed)
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    questions_frequency = defaultdict(int)

    for item_data in data_uniformed:
        for question_id in item_data["question_seq"]:
            questions_frequency[question_id] += 1

    questions2drop = set()
    for q_id in questions_frequency.keys():
        if questions_frequency[q_id] < num2drop:
            questions2drop.add(q_id)

    data_dropped = []
    num_drop_interactions = 0
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            if q_id in questions2drop:
                num_drop_interactions += 1
                continue
            for k in seq_keys:
                item_data_new[k].append(item_data[k][i])
        item_data_new["seq_len"] = len(item_data_new["question_seq"])
        data_dropped.append(item_data_new)

    return data_dropped
