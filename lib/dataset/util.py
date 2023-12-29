from collections import defaultdict

from ..util.parse import get_keys_from_uniform
from ..util.data import data_agg_question


def data_kt2srs(data_uniformed, data_type):
    if data_type == "multi_concept":
        data_uniformed = data_agg_question(data_uniformed)

    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    data_transformed = []
    for i, item_data in enumerate(data_uniformed):
        for j in range(1, item_data["seq_len"]):
            item_data_new = {
                "target_seq_id": i,
                "target_seq_len": j
            }
            for k in seq_keys:
                if k == "concept_seq":
                    item_data_new["target_concept"] = item_data[k][j]
                if k == "question_seq":
                    item_data_new["target_question"] = item_data[k][j]
                if k == "correct_seq":
                    item_data_new["target_correct"] = item_data[k][j]
                if k == "time_seq":
                    item_data_new["target_time"] = item_data[k][j]
                if k == "use_time_seq":
                    item_data_new["target_use_time"] = item_data[k][j]
            data_transformed.append(item_data_new)
    return data_transformed


def parse_difficulty(data_uniformed, data_type, num_min_question, num_min_concept, num_question_diff, num_concept_diff):
    questions_frequency, concepts_frequency = defaultdict(int), defaultdict(int)
    questions_accuracy, concepts_accuracy = defaultdict(int), defaultdict(int)
    if data_type == "single_concept":
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                c_id = item_data["concept_seq"][i]
                questions_frequency[q_id] += 1
                concepts_frequency[c_id] += 1
                if item_data["correct_seq"][i] == 1:
                    questions_accuracy[q_id] += 1
                    concepts_accuracy[c_id] += 1
        for q_id in questions_frequency.keys():
            if questions_frequency[q_id] < num_min_question:
                questions_accuracy[q_id] = num_question_diff
            else:
                questions_accuracy[q_id] = int((num_question_diff - 1) * questions_accuracy[q_id] / questions_frequency[q_id])
        for c_id in concepts_frequency.keys():
            if concepts_frequency[c_id] < num_min_concept:
                concepts_accuracy[c_id] = num_concept_diff
            else:
                concepts_accuracy[c_id] = int((num_concept_diff - 1) * concepts_accuracy[c_id] / concepts_frequency[c_id])
    else:
        raise NotImplementedError()

    return questions_accuracy, concepts_accuracy


def parse_long_tail(data_uniformed, data_type, head_question_threshold, head_seq_len, min_context_seq_len):
    # question context，即每道习题对应的序列，格式为{q_id: [{seq_id, seq_len, correct}, ...], ...}
    question_context = defaultdict(list)
    questions_frequency = defaultdict(int)
    head_seqs = []
    if data_type == "single_concept":
        for seq_id, item_data in enumerate(data_uniformed):
            if item_data["seq_len"] > head_seq_len:
                head_seqs.append(seq_id)
            if min_context_seq_len >= item_data["seq_len"]:
                continue
            for i in range(min_context_seq_len, item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                correct = item_data["correct_seq"][i]
                questions_frequency[q_id] += 1
                q_context = {"seq_id": seq_id, "seq_len": i, "correct": correct}
                question_context[q_id].append(q_context)
    else:
        raise NotImplementedError()

    question_list = list(questions_frequency.items())
    question_list = sorted(question_list, key=lambda x: x[1])
    question_list = list(map(lambda x: x[0], question_list))
    num_question = len(question_list)
    head_questions = question_list[int(num_question * head_question_threshold):]

    return question_context, head_questions, head_seqs


