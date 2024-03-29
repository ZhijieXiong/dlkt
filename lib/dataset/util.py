from collections import defaultdict

from ..util.parse import get_keys_from_uniform


def data_kt2srs(data_uniformed):
    table4srs = {
        "mask_seq": "target_mask",
        "concept_seq": "target_concept",
        "question_seq": "target_question",
        "correct_seq": "target_correct",
        "time_seq": "target_time",
        "interval_time_seq": "target_interval_time",
        "use_time_seq": "target_use_time",
        "question_diff_seq": "target_question_diff",
        "concept_diff_seq": "target_concept_diff",
        "age_seq": "target_age"
    }
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    data_transformed = []
    for i, item_data in enumerate(data_uniformed):
        for j in range(1, item_data["seq_len"]):
            item_data_new = {
                "target_seq_id": i,
                "target_seq_len": j
            }
            for k in seq_keys:
                item_data_new[table4srs[k]] = item_data[k][j]
            data_transformed.append(item_data_new)
    return data_transformed


def parse_difficulty(data_uniformed, params, objects):
    data_type = params["data_type"]
    num_min_question = params["num_min_question"]
    num_min_concept = params["num_min_concept"]
    num_question_diff = params["num_question_diff"]
    num_concept_diff = params["num_concept_diff"]
    num_concept = params["num_concept"]
    num_question = params["num_question"]

    questions_frequency, concepts_frequency = defaultdict(int), defaultdict(int)
    questions_accuracy, concepts_accuracy = defaultdict(int), defaultdict(int)
    # 用于给统计信息不足的知识点和习题赋值难度
    num_few_shot_q = 0
    num_few_shot_c = 0
    if data_type == "single_concept":
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                c_id = item_data["concept_seq"][i]
                questions_frequency[q_id] += 1
                concepts_frequency[c_id] += 1
                questions_accuracy[q_id] += item_data["correct_seq"][i]
                concepts_accuracy[c_id] += item_data["correct_seq"][i]
    elif data_type == "only_question":
        question2concept = objects["question2concept"]
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                questions_frequency[q_id] += 1
                c_ids = question2concept[q_id]
                for c_id in c_ids:
                    concepts_frequency[c_id] += 1
                questions_accuracy[q_id] += item_data["correct_seq"][i]
                for c_id in c_ids:
                    concepts_accuracy[c_id] += item_data["correct_seq"][i]
    else:
        raise NotImplementedError()

    for q_id in range(num_question):
        if questions_frequency[q_id] < num_min_question:
            questions_accuracy[q_id] = num_question_diff + num_few_shot_q
            num_few_shot_q += 1
        else:
            questions_accuracy[q_id] = int(
                (num_question_diff - 1) * questions_accuracy[q_id] / questions_frequency[q_id])
    for c_id in range(num_concept):
        if concepts_frequency[c_id] < num_min_concept:
            concepts_accuracy[c_id] = num_concept_diff + num_few_shot_c
            num_few_shot_c += 1
        else:
            concepts_accuracy[c_id] = int((num_concept_diff - 1) * concepts_accuracy[c_id] / concepts_frequency[c_id])

    return questions_accuracy, concepts_accuracy


def parse_long_tail(data_uniformed, params):
    """

    :param data_uniformed:
    :param params:
    :return:
    """
    data_type = params["data_type"]
    head_question_threshold = params.get("head_question_threshold", 0.8)
    head_seq_len = params.get("head_seq_len", 20)
    min_context_seq_len = params.get("min_context_seq_len", 10)

    # question context，即每道习题对应的序列，格式为{q_id: [{seq_id, seq_len, correct}, ...], ...}
    question_context = defaultdict(list)
    questions_frequency = defaultdict(int)
    head_seqs = []
    tail_questions = []
    if data_type in ["single_concept", "only_question"]:
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
        # multi concept做不了这个
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


def parse4dataset_enhanced(data_uniformed, question2concept, concept2question, enhance_params):
    """
    将所有习题（同一知识点或者有相同知识点）分为easy、middle、hard、unknown\n
    :param data_uniformed:
    :param question2concept:
    :param concept2question:
    :param enhance_params:
    :return:
    """
    # 需要生成两个dict，一个是知识点，形式为{c0: {"easy": [q0, q1, ...], "middle": [], "hard": [], "unknown":[]}, ...}
    # 其中每个难度（easy、middle、hard）下的习题列表都是按难度顺序排列，由易到难
    # 另一个dict是习题，形式为{q0: [(c0, "easy", 0), (c1, "easy", 0), ...]}，表示习题q0是知识点c0下的easy题，并且在easy题中排列第0
    # 用绝对值区分习题难度档次，如正确率小于0.3为难题，大于0.85为简单题，要考虑数据集整体正确率来确定
    num_min_question4diff = enhance_params.get("num_min_question4diff", 100)
    num_few_shot = enhance_params.get("num_few_shot", 5)
    hard_acc = enhance_params.get("hard_acc", 0.3)
    easy_acc = enhance_params.get("easy_acc", 0.85)

    question_frequency = {}
    question_accuracy = {}
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
