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
    tail_questions = []
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

    for q_id, fre in questions_frequency.items():
        if fre < 5:
            tail_questions.append(q_id)

    question_list = list(questions_frequency.items())
    question_list = sorted(question_list, key=lambda x: x[1])
    question_list = list(map(lambda x: x[0], question_list))
    num_question = len(question_list)
    head_questions = question_list[int(num_question * head_question_threshold):]

    return question_context, head_questions, tail_questions, head_seqs


def parse_low_fre_question(data_uniformed, data_type, num_few_shot, num_question):
    questions_frequency = defaultdict(int)
    few_shot_questions = []
    if data_type == "single_concept":
        for seq_id, item_data in enumerate(data_uniformed):
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                questions_frequency[q_id] += 1

    for q_id, fre in questions_frequency.items():
        if fre < num_few_shot:
            few_shot_questions.append(q_id)

    zero_shot_questions = list(set(range(num_question)) - set(questions_frequency.keys()))

    return zero_shot_questions, few_shot_questions


def parse4dataset_enhanced(data_uniformed, data_type, num_min_question4diff, num_few_shot, question2concept, concept2question, hard_acc=0.4, easy_acc=0.8):
    """
    将所有习题（同一知识点或者有相同知识点）分为easy、middle、hard、unknown\n
    :param data_uniformed:
    :param data_type:
    :param num_min_question4diff:
    :param num_few_shot:
    :param question2concept:
    :param concept2question:
    :param hard_acc:
    :param easy_acc:
    :return:
    """
    # 需要生成两个dict，一个是知识点，形式为{c0: {"easy": [q0, q1, ...], "middle": [], "hard": [], "unknown":[]}, ...}
    # 其中每个难度下的习题列表都是按难度顺序排列，由易到难，除了unknown
    # 另一个dict是习题，形式为{q0: [(c0, "easy", 0), (c1, "easy", 0), ...]}，表示习题q0是知识点c0下的easy题，并且在easy题中排列第0
    # 用绝对值区分习题难度档次，如正确率小于0.3为难题，大于0.8为简单题，要考虑数据集整体正确率来确定
    question_frequency = {}
    question_accuracy = {}
    if data_type in ["single_concept", "only_question"]:
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                question_frequency.setdefault(q_id, 0)
                question_accuracy.setdefault(q_id, 0)
                question_frequency[q_id] += 1
                if item_data["correct_seq"][i] == 1:
                    question_accuracy[q_id] += 1
        for q_id in question_frequency.keys():
            if question_frequency[q_id] >= num_min_question4diff:
                question_accuracy[q_id] = question_accuracy[q_id] / question_frequency[q_id]
    else:
        raise NotImplementedError()

    concept_dict = {}
    question_dict = {}
    for c_id in range(len(concept2question)):
        concept_dict[c_id] = {
            "easy": [],
            "middle": [],
            "hard": [],
            "zero_shot": [],
            "few_shot": [],
            "middle_fre": []
        }
        correspond_questions = concept2question[c_id]
        for q_id in correspond_questions:
            correspond_concepts = question2concept[q_id]
            correspond_concepts = list(map(str, sorted(correspond_concepts)))
            question_dict[q_id] = (",".join(correspond_concepts), [])
            if not question_frequency.get(q_id, False):
                concept_dict[c_id]["zero_shot"].append(q_id)
                question_dict[q_id][1].append([c_id, "zero_shot"])
                continue
            q_count = question_frequency[q_id]
            q_acc = question_accuracy[q_id]
            if q_count <= num_few_shot:
                concept_dict[c_id]["few_shot"].append(q_id)
                question_dict[q_id][1].append([c_id, "few_shot"])
            elif q_count < num_min_question4diff:
                concept_dict[c_id]["middle_fre"].append(q_id)
                question_dict[q_id][1].append([c_id, "middle_fre"])
            elif q_acc < hard_acc:
                concept_dict[c_id]["hard"].append((q_id, q_acc))
                question_dict[q_id][1].append([c_id, "hard"])
            elif q_acc > easy_acc:
                concept_dict[c_id]["easy"].append((q_id, q_acc))
                question_dict[q_id][1].append([c_id, "easy"])
            else:
                concept_dict[c_id]["middle"].append((q_id, q_acc))
                question_dict[q_id][1].append([c_id, "middle"])

    # 对每个档次下的习题排序
    for c_id in concept_dict.keys():
        for k in ["easy", "middle", "hard"]:
            questions = concept_dict[c_id][k]
            concept_dict[c_id][k] = list(map(
                lambda x: x[0],
                sorted(questions, key=lambda x: x[1], reverse=True)
            ))

    for q_id in question_dict.keys():
        for c_pair in question_dict[q_id][1]:
            c_id = c_pair[0]
            k = c_pair[1]
            if k in ["easy", "middle", "hard"]:
                c_pair.append(concept_dict[c_id][k].index(q_id))
            else:
                c_pair.append(0)

    return concept_dict, question_dict

