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


# 辅助函数，计算统计信息
def static_func(data_uniformed):
    num_interaction = 0
    num_student = len(data_uniformed)
