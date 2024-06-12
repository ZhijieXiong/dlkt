import math


def discount(correct_seq, seq_len, max_seq_len):
    # 生成weight seq
    w_seq = [1]
    # 双指针
    p1, p2 = 1, 1
    count_same = 0
    while p2 < seq_len:
        if correct_seq[p1] == correct_seq[p2]:
            count_same += 1
        else:
            p1 = p2
            count_same = 0
        if count_same > 3:
            w_seq.append(0.99 ** (count_same - 3))
        else:
            w_seq.append(1)
        p2 += 1
    w_seq += [1] * (max_seq_len - len(w_seq))

    return w_seq


def highlight_tail(w, question_seq, train_statics, seq_len, max_seq_len):
    w_seq = [1]
    question_low_fre = train_statics["question_low_fre"]
    for q in question_seq[1:seq_len]:
        if q in question_low_fre:
            w_seq.append(w)
        else:
            w_seq.append(1)
    w_seq += [1] * (max_seq_len - len(w_seq))

    return w_seq


def IPS_weight(item_data, question2concept, IPS_min, IPS_his_seq_len):
    question_seq = item_data["question_seq"]
    concept_context = list(map(lambda q: question2concept[q], question_seq))
    correct_seq = item_data["correct_seq"]
    max_seq_len = len(question_seq)
    seq_len = item_data["seq_len"]

    if seq_len <= IPS_his_seq_len:
        return [1.0] * seq_len + [0.] * (max_seq_len - seq_len)

    weight_seq = [1.0] * IPS_his_seq_len
    for i in range(IPS_his_seq_len, seq_len):
        correct_context = correct_seq[i-IPS_his_seq_len:i]
        context_concepts = set([c_id for c_ids in concept_context[i - IPS_his_seq_len:i] for c_id in c_ids])
        current_concept = set([c_id for c_id in concept_context[i]])

        seq_accuracy = sum(correct_context) / IPS_his_seq_len

        if bool(context_concepts & current_concept):
            weight = 1.0
        else:
            weight = IPS_min + math.fabs(seq_accuracy - 0.5) * (1 - IPS_min) / 0.5

        weight_seq.append(weight)
    weight_seq += [0] * (max_seq_len - seq_len)

    return weight_seq
