import json
import ast
import os
import torch
import numpy as np


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
            current_key = keys[int(i % num_key)]
            if current_key in ["time_factor_seq", "hint_factor_seq", "attempt_factor_seq", "correct_float_seq"]:
                line_content = list(map(float, line_str.strip().split(",")))
            else:
                line_content = list(map(int, line_str.strip().split(",")))
            if len(line_content) == 1:
                # 说明是序列级别的特征，即user id、seq len、segment index等等
                item_data[current_key] = line_content[0]
            else:
                # 说明是interaction级别的特征，即question id等等
                item_data[current_key] = line_content
            if i % num_key == (num_key - 1):
                data.append(item_data)

    return data


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def is_valid_eval_string(in_str):
    try:
        ast.literal_eval(in_str)
        return True
    except (SyntaxError, ValueError):
        return False


def str_dict2params_tool(param):
    if is_valid_eval_string(param):
        return eval(param)
    else:
        return param


def str_dict2params(str_dict):
    params = {}
    for k, v in str_dict.items():
        if type(v) is not dict:
            params[k] = str_dict2params_tool(v)
        else:
            params[k] = str_dict2params(v)
    return params


def load_q_table(q_table_path):
    Q_table = np.load(q_table_path)
    return Q_table


def concept2question_from_Q(Q_table):
    # 将Q table转换为{concept_id1: [question_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[:, i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[1])}
    return result


def question2concept_from_Q(Q_table):
    # 将Q table转换为{question_id1: [concept_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[0])}
    return result


def parse_Q_table(Q_table, device):
    """
    生成多知识点embedding融合需要的数据
    :return:
    """
    question2concept_table = []
    question2concept_mask_table = []
    num_max_c_in_q = np.max(np.sum(Q_table, axis=1))
    num_question = Q_table.shape[0]
    for i in range(num_question):
        cs = np.argwhere(Q_table[i] == 1).reshape(-1).tolist()
        pad_len = num_max_c_in_q - len(cs)
        question2concept_table.append(cs + [0] * pad_len)
        question2concept_mask_table.append([1] * len(cs) + [0] * pad_len)
    question2concept_table = torch.tensor(question2concept_table).long().to(device)
    question2concept_mask_table = torch.tensor(question2concept_mask_table).long().to(device)
    return question2concept_table, question2concept_mask_table, num_max_c_in_q


def get_global_objects_data(q_table_path, device):
    objects_data = {}
    Q_table = load_q_table(q_table_path)
    objects_data["Q_table"] = Q_table
    objects_data["Q_table_tensor"] = torch.from_numpy(Q_table).long().to(device)
    objects_data["question2concept"] = question2concept_from_Q(Q_table)
    objects_data["concept2question"] = concept2question_from_Q(Q_table)
    q2c_table, q2c_mask_table, num_max_concept = parse_Q_table(Q_table, device)
    objects_data["q2c_table"] = q2c_table
    objects_data["q2c_mask_table"] = q2c_mask_table
    objects_data["num_max_concept"] = num_max_concept
    return objects_data
