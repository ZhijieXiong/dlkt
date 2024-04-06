import numpy as np
from collections import Counter
from copy import deepcopy

from .data import dataset_agg_concept, dataset_delete_pad


def cal_frequency(data_uniformed, num_item, target="question"):
    target_seq = "concept_seq" if target == "concept" else "question_seq"
    item_seqs = list(map(lambda item_data: item_data[target_seq], data_uniformed))
    items = []
    for item_seq in item_seqs:
        items += item_seq
    item_frequency = Counter(items)

    for item in range(num_item):
        if item not in item_frequency.keys():
            item_frequency[item] = 0

    return {item_id: item_frequency[item_id] for item_id in range(num_item)}


def cal_accuracy(data_uniformed, num_item, target="question"):
    """
    未出现的正确率记为-1
    :param data_uniformed:
    :param num_item:
    :param target:
    :return:
    """
    target_seq = "concept_seq" if target == "concept" else "question_seq"
    item_acc_dict = {i: -1 for i in range(num_item)}

    count = {i: 0 for i in range(num_item)}
    correct = {i: 0 for i in range(num_item)}
    for item_data in data_uniformed:
        for item_id, c in zip(item_data[target_seq], item_data["correct_seq"]):
            count[item_id] += 1
            if c == 1:
                correct[item_id] += 1

    for item_id in range(num_item):
        if count[item_id] != 0:
            item_acc_dict[item_id] = correct[item_id] / count[item_id]

    return item_acc_dict


def cal_acc_overall(data_uniformed):
    num_right = 0
    num_all = 0
    for item_data in data_uniformed:
        seq_len = item_data["seq_len"]
        num_right += sum(item_data["correct_seq"][:seq_len])
        num_all += seq_len

    # 总体正确率
    acc_overall = round(num_right / num_all * 100, 2)

    return acc_overall


def dataset_basic_statics(data_uniformed, data_type, question2concept, num_question=0, num_concept=0):
    data_uniformed = dataset_delete_pad(data_uniformed)

    if data_type == "only_question":
        acc_overall = cal_acc_overall(data_uniformed)
        question_fre = cal_frequency(data_uniformed, num_question, "question")
        question_acc = cal_accuracy(data_uniformed, num_question, "question")
        # 统计知识点正确率
        for item_data in data_uniformed:
            item_data["concept_seq"] = []
            item_data["correct_seq4concept"] = []
            item_data["correct_seq_backup"] = deepcopy(item_data["correct_seq"])
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                correct = item_data["correct_seq"][i]
                c_ids = question2concept[q_id]
                item_data["concept_seq"] += c_ids
                item_data["correct_seq4concept"] += [correct] * len(c_ids)
            item_data["seq_len"] = len(item_data["concept_seq"])
            item_data["correct_seq"] = item_data.pop("correct_seq4concept")
        concept_fre = cal_frequency(data_uniformed, num_concept, "concept")
        concept_acc = cal_accuracy(data_uniformed, num_concept, "concept")
    elif data_type == "multi_concept":
        data_only_question = dataset_agg_concept(data_uniformed)
        acc_overall = cal_acc_overall(data_only_question)
        question_fre = cal_frequency(data_only_question, num_question, "question")
        concept_fre = cal_frequency(data_uniformed, num_concept, "concept")
        question_acc = cal_accuracy(data_only_question, num_question, "question")
        concept_acc = cal_accuracy(data_uniformed, num_concept, "concept")
    else:
        acc_overall = cal_acc_overall(data_uniformed)
        question_fre = cal_frequency(data_uniformed, num_question, "question")
        concept_fre = cal_frequency(data_uniformed, num_concept, "concept")
        question_acc = cal_accuracy(data_uniformed, num_question, "question")
        concept_acc = cal_accuracy(data_uniformed, num_concept, "concept")

    return {
        "acc_overall": acc_overall,
        "question_fre": question_fre,
        "question_acc": question_acc,
        "concept_fre": concept_fre,
        "concept_acc": concept_acc
    }


def cal_propensity(data_uniformed, num_item, data_type, target_item="concept"):
    """
    计算数据集中question或者concept的频率（DROS使用的公式）

    :param data_uniformed:
    :param num_item: num_item is num_question if target_item=="concept", else num_concept
    :param data_type:
    :param target_item:
    :return:
    """
    if target_item == "concept":
        if data_type == "only_question":
            pass
        else:
            item_seqs = list(map(
                lambda item: item["concept_seq"][:item["seq_len"]],
                data_uniformed
            ))
    else:
        if data_type == "multi_concept":
            pass
        else:
            item_seqs = list(map(
                lambda item: item["question_seq"][:item["seq_len"]],
                data_uniformed
            ))

    items = []
    for item_seq in item_seqs:
        items += item_seq
    freq = Counter(items)
    for i in range(num_item):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(num_item)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)

    return ps

