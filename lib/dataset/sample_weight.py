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
