from collections import Counter

from .data import dataset_agg_concept, dataset_delete_pad


def cal_frequency(data_uniformed, num_item, target="question"):
    target_seq = "concept_seq" if target == "concept" else "question_seq"
    item_seqs = list(map(lambda item_data: item_data[target_seq], data_uniformed))
    items = []
    for question_seq in item_seqs:
        items += question_seq
    item_frequency = Counter(items)

    for item in range(num_item):
        if item not in item_frequency.keys():
            item_frequency[item] = 0

    return {item_id: item_frequency[item_id] for item_id in range(num_item)}


def cal_accuracy(data_uniformed, num_item, target="question"):
    target_seq = "concept_seq" if target == "concept" else "question_seq"
    item_difficulty = {i: -1 for i in range(num_item)}

    count = {i: 0 for i in range(num_item)}
    correct = {i: 0 for i in range(num_item)}
    for item_data in data_uniformed:
        for item_id, c in zip(item_data[target_seq], item_data["correct_seq"]):
            count[item_id] += 1
            if c == 1:
                correct[item_id] += 1

    for item_id in range(num_item):
        if count[item_id] != 0:
            item_difficulty[item_id] = correct[item_id] / count[item_id]

    return item_difficulty


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


def dataset_basic_statics(data_uniformed, data_type, num_question=0, num_concept=0):
    data_uniformed = dataset_delete_pad(data_uniformed)

    if data_type == "only_question":
        acc_overall = cal_acc_overall(data_uniformed)
        question_fre = cal_frequency(data_uniformed, num_question, "question")
        question_acc = cal_accuracy(data_uniformed, num_question, "question")
        concept_fre = -1
        concept_acc = -1
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