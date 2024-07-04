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


def IPS_seq_weight(item_data, IPS_min, IPS_his_seq_len):
    question_seq = item_data["question_seq"]
    correct_seq = item_data["correct_seq"]
    max_seq_len = len(question_seq)
    seq_len = item_data["seq_len"]

    if seq_len <= IPS_his_seq_len:
        return [1.0] * seq_len + [0.] * (max_seq_len - seq_len)

    weight_seq = [1.0] * IPS_his_seq_len
    for i in range(IPS_his_seq_len, seq_len):
        correct_context = correct_seq[i-IPS_his_seq_len:i]
        seq_accuracy = sum(correct_context) / IPS_his_seq_len
        label_current = correct_seq[i]

        if ((seq_accuracy <= 0.5) and (label_current == 1)) or ((seq_accuracy >= 0.5) and (label_current == 0)):
            weight = 1.0
        else:
            weight = IPS_min + math.fabs(seq_accuracy - 0.5) * (1 - IPS_min) / 0.5

        weight_seq.append(weight)
    weight_seq += [0] * (max_seq_len - seq_len)

    return weight_seq


def IPS_question_weight(item_data, statics_train, IPS_min):
    question_seq = item_data["question_seq"]
    correct_seq = item_data["correct_seq"]
    max_seq_len = len(question_seq)
    seq_len = item_data["seq_len"]

    weight_seq = []
    for i in range(seq_len):
        q_id = question_seq[i]
        q_acc_statics = statics_train["question_acc"][q_id]
        label_current = correct_seq[i]

        if (q_acc_statics < 0) or \
                ((q_acc_statics <= 0.5) and (label_current == 1)) or \
                ((q_acc_statics >= 0.5) and (label_current == 0)):
            weight = 1.0
        else:
            weight = IPS_min + math.fabs(q_acc_statics - 0.5) * (1 - IPS_min) / 0.5

        weight_seq.append(weight)
    weight_seq += [0] * (max_seq_len - seq_len)

    return weight_seq


def IPS_double_weight(item_data, statics_train, IPS_min, IPS_his_seq_len):
    question_seq = item_data["question_seq"]
    correct_seq = item_data["correct_seq"]
    max_seq_len = len(question_seq)
    seq_len = item_data["seq_len"]

    weight_seq = []
    for i in range(seq_len):
        q_id = question_seq[i]
        q_acc_statics = statics_train["question_acc"][q_id]
        label_current = correct_seq[i]

        if (q_acc_statics < 0) or \
                ((q_acc_statics <= 0.5) and (label_current == 1)) or \
                ((q_acc_statics >= 0.5) and (label_current == 0)):
            weight_1 = 1.0
        else:
            weight_1 = IPS_min + math.fabs(q_acc_statics - 0.5) * (1 - IPS_min) / 0.5

        if i < IPS_his_seq_len:
            weight_2 = 1.0
        else:
            correct_context = correct_seq[i - IPS_his_seq_len:i]
            seq_accuracy = sum(correct_context) / IPS_his_seq_len

            if ((seq_accuracy <= 0.5) and (label_current == 1)) or ((seq_accuracy >= 0.5) and (label_current == 0)):
                weight_2 = 1.0
            else:
                weight_2 = IPS_min + math.fabs(seq_accuracy - 0.5) * (1 - IPS_min) / 0.5

        weight_seq.append((weight_1 + weight_2) / 2)
    weight_seq += [0] * (max_seq_len - seq_len)

    return weight_seq
