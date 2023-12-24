import torch
import numpy as np
from sklearn import metrics


def record_dis4seq_len(ground_truth, prediction, mask):
    batch_size, max_seq_len = ground_truth.shape[0], ground_truth.shape[1]
    label_dis = [[] for _ in range(max_seq_len)]
    score_dis = [[] for _ in range(max_seq_len)]
    seqs_len = torch.sum(mask, dim=1).tolist()
    g = ground_truth.tolist()
    p = prediction.tolist()
    for i, seq_len in enumerate(seqs_len):
        for j in range(seq_len):
            label_dis[j].append(g[i][j])
            score_dis[j].append(p[i][j])
    return label_dis, score_dis


def evaluate4seq_len(label_dis, score_dis, split_len):
    label_dis4len = [[] for _ in range(len(split_len) - 1)]
    score_dis4len = [[] for _ in range(len(split_len) - 1)]
    indices4len = []
    for i in range(len(split_len)):
        if i == len(split_len) - 1:
            break
        indices4len.append((split_len[i], split_len[i + 1]))
    for i, indices in enumerate(indices4len):
        index_s, index_e = indices
        tmp_label_dis = label_dis[index_s: index_e]
        tmp_score_dis = score_dis[index_s: index_e]
        label_dis4len[i] += [item for item_list in tmp_label_dis for item in item_list]
        score_dis4len[i] += [item for item_list in tmp_score_dis for item in item_list]
    print("split by seq length")
    for i in range(len(label_dis4len)):
        if len(label_dis4len[i]) == 0:
            continue
        g = np.array(label_dis4len[i])
        p = np.array(score_dis4len[i])
        p_label = [1 if _ >= 0.5 else 0 for _ in p]
        answer_acc = g.sum() / len(g)
        print(f"({indices4len[i][0]:<3}, {indices4len[i][1]:<3}), num of samples is {g.size:<10}, "
              f"acc of answer is {answer_acc*100:<4.3}%: "
              f"auc is {metrics.roc_auc_score(y_true=g, y_score=p):<9.5}, "
              f"acc is {metrics.accuracy_score(g, p_label):<9.5}")


def evaluate_bias(result_all_batch, n):
    """
    在预测错的item上往回看1,2,3,...,n个step，看看和之前的标签（做对或者做错）是否有关联
    :param result_all_batch:
    :param n:
    :return:
    """
    predict_wrong = []
    for batch in result_all_batch:
        for label_seq, predict_score_seq, correct_seq, mask_seq in zip(batch["label"],
                                                                       batch["predict_score"],
                                                                       batch["correct_seq"],
                                                                       batch["mask_seq"]):
            if mask_seq[-1] != 1:
                seq_len = mask_seq.index(0)
            else:
                seq_len = len(mask_seq)
            for i, _ in enumerate(mask_seq[1:seq_len]):
                label = label_seq[i]
                predict_label = 1 if predict_score_seq[i] > 0.5 else 0
                if label != predict_label:
                    case_wrong = {}
                    for j in range(1, n):
                        past_labels = correct_seq[max(0, i+1-j):i+1]
                        case_wrong[j] = (sum(past_labels) if (predict_label == 1) else (len(past_labels) - sum(past_labels))) / len(past_labels)

    return predict_wrong
