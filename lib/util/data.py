import os
import json

import numpy
import numpy as np


def write2file(data, data_path):
    # seq_feature_keys表示序列级别的特征，如user_id, seq_len
    # interaction_feature_keys表示交互级别的特征，如question_id, concept_id
    seq_keys = []
    interaction_keys = []
    for key in data[0].keys():
        if type(data[0][key]) == list:
            interaction_keys.append(key)
        else:
            seq_keys.append(key)
    with open(data_path, "w") as f:
        first_line = ",".join(seq_keys) + ";" + ",".join(interaction_keys) + "\n"
        f.write(first_line)
        for item_data in data:
            for seq_key in seq_keys:
                f.write(f"{item_data[seq_key]}\n")
            for interaction_key in interaction_keys:
                f.write(",".join(map(str, item_data[interaction_key])) + "\n")


def read_preprocessed_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        seq_interaction_keys_str = first_line.split(";")
        seq_keys_str = seq_interaction_keys_str[0].strip()
        interaction_keys_str = seq_interaction_keys_str[1].strip()
        seq_keys = seq_keys_str.split(",")
        interaction_keys = interaction_keys_str.split(",")
        keys = seq_keys + interaction_keys
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
