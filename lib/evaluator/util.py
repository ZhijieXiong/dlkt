import torch
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error


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

    return label_dis4len, score_dis4len, indices4len


def get_seq_biased_point(all_batch, previous_seq_len, seq_most_accuracy):
    """
    返回每条序列中满足以下条件的点：\n
    1. 该点的context seq len为previous_seq_len，即从每条序列的第previous_seq_len个点开始\n
    2. 该点的context seq正确率大于(1 - seq_most_accuracy)，且做错，或者小于seq_most_accuracy，且做对\n
    :param all_batch:
    :param previous_seq_len:
    :param seq_most_accuracy:
    :return:
    """
    result = {
        "high_acc_but_wrong": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        },
        "low_acc_but_right": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        }
    }
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["predict_score_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, predict_score_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq[previous_seq_len:]):
                i += previous_seq_len
                if m == 0:
                    break

                context_label = label_seq[i-previous_seq_len:i]
                context_accuracy = sum(context_label) / len(context_label)

                if (context_accuracy <= seq_most_accuracy) and (label_seq[i] == 1):
                    result["low_acc_but_right"]["question"].append(question_seq[i])
                    result["low_acc_but_right"]["predict_score"].append(predict_score_seq[i])
                    result["low_acc_but_right"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                elif (context_accuracy >= (1 - seq_most_accuracy)) and (label_seq[i] == 0):
                    result["high_acc_but_wrong"]["question"].append(question_seq[i])
                    result["high_acc_but_wrong"]["predict_score"].append(predict_score_seq[i])
                    result["high_acc_but_wrong"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                else:
                    pass

    return result


def evaluate_bias(seq_biased_point):
    """
    评估数据集中存在序列偏差的点
    :param seq_biased_point:
    :return:
    """
    # todo: 如果seq_biased_label全为0或者全为1，则不能计算AUC，这个还未处理
    seq_biased_label = [1] * len(seq_biased_point["low_acc_but_right"]["predict_score"]) + \
                       [0] * len(seq_biased_point["high_acc_but_wrong"]["predict_score"])
    seq_biased_predict_score = seq_biased_point["low_acc_but_right"]["predict_score"] + \
                               seq_biased_point["high_acc_but_wrong"]["predict_score"]
    seq_biased_predict_label = seq_biased_point["low_acc_but_right"]["predict_label"] + \
                               seq_biased_point["high_acc_but_wrong"]["predict_label"]
    result = {
        "num_sample": len(seq_biased_label),
        "AUC": roc_auc_score(y_true=seq_biased_label, y_score=seq_biased_predict_score),
        "ACC": accuracy_score(y_true=seq_biased_label, y_pred=seq_biased_predict_label),
        "RMSE": mean_squared_error(y_true=seq_biased_label, y_pred=seq_biased_predict_score),
        "MAE": mean_absolute_error(y_true=seq_biased_label, y_pred=seq_biased_predict_score)
    }

    return result


def evaluate_double_bias(seq_biased_point, statics_train):
    """
    评估数据集中存在序列偏差和习题偏差问题的点
    :param seq_biased_point:
    :param statics_train:
    :return:
    """
    # todo: 如果seq_biased_label全为0或者全为1，则不能计算AUC，这个还未处理
    double_biased_label = []
    double_biased_predict_score = []
    double_biased_predict_label = []
    for q_id, p_score, p_label in zip(seq_biased_point["low_acc_but_right"]["question"],
                                      seq_biased_point["low_acc_but_right"]["predict_score"],
                                      seq_biased_point["low_acc_but_right"]["predict_label"]):
        if q_id in statics_train["question_low_acc"]:
            double_biased_label.append(1)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    for q_id, p_score, p_label in zip(seq_biased_point["high_acc_but_wrong"]["question"],
                                      seq_biased_point["high_acc_but_wrong"]["predict_score"],
                                      seq_biased_point["high_acc_but_wrong"]["predict_label"]):
        if q_id in statics_train["question_high_acc"]:
            double_biased_label.append(0)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    return {
        "num_sample": len(double_biased_label),
        "AUC": roc_auc_score(y_true=double_biased_label, y_score=double_biased_predict_score),
        "ACC": accuracy_score(y_true=double_biased_label, y_pred=double_biased_predict_label),
        "RMSE": mean_squared_error(y_true=double_biased_label, y_pred=double_biased_predict_score),
        "MAE": mean_absolute_error(y_true=double_biased_label, y_pred=double_biased_predict_score)
    }
