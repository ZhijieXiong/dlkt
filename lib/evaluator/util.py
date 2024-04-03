import numpy as np
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


def get_performance_no_error(predict_score, predict_label, true_label):
    if len(predict_label) == 0:
        return {
            "num_sample": 0,
            "AUC": -1.,
            "ACC": -1.,
            "RMSE": -1.,
            "MAE": -1.
        }

    try:
        AUC = roc_auc_score(y_true=true_label, y_score=predict_score)
    except ValueError:
        AUC = -1.

    result = {
        "num_sample": len(true_label),
        "AUC": AUC,
        "ACC": accuracy_score(y_true=true_label, y_pred=predict_label),
        "RMSE": mean_squared_error(y_true=true_label, y_pred=predict_score),
        "MAE": mean_absolute_error(y_true=true_label, y_pred=predict_score)
    }

    return result


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


def get_question_biased_point(all_batch, statics_train, most_accuracy):
    """
    返回满足以下条件的点：该习题是高正确率习题，但是做错，或者，该习题是低正确率习题，但是做对\n
    :param all_batch:
    :param statics_train:
    :param most_accuracy:
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
            for i, m in enumerate(mask_seq):
                if m == 0:
                    break
                q_id = question_seq[i]
                q_acc_statics = statics_train["question_acc"][q_id]
                label = label_seq[i]
                if q_acc_statics < 0:
                    continue
                if (q_acc_statics > (1 - most_accuracy)) and (label == 0):
                    result["high_acc_but_wrong"]["question"].append(q_id)
                    result["high_acc_but_wrong"]["predict_score"].append(predict_score_seq[i])
                    result["high_acc_but_wrong"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                if (q_acc_statics < most_accuracy) and (label == 1):
                    result["high_acc_but_wrong"]["question"].append(q_id)
                    result["high_acc_but_wrong"]["predict_score"].append(predict_score_seq[i])
                    result["high_acc_but_wrong"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)

    return result


def evaluate_bias(seq_biased_point):
    """
    评估数据集中存在序列偏差的点
    :param seq_biased_point:
    :return:
    """
    seq_biased_label = [1] * len(seq_biased_point["low_acc_but_right"]["predict_score"]) + \
                       [0] * len(seq_biased_point["high_acc_but_wrong"]["predict_score"])
    seq_biased_predict_score = seq_biased_point["low_acc_but_right"]["predict_score"] + \
                               seq_biased_point["high_acc_but_wrong"]["predict_score"]
    seq_biased_predict_label = seq_biased_point["low_acc_but_right"]["predict_label"] + \
                               seq_biased_point["high_acc_but_wrong"]["predict_label"]

    return get_performance_no_error(seq_biased_predict_score, seq_biased_predict_label, seq_biased_label)


def evaluate_double_bias(seq_biased_point, statics_train, most_accuracy):
    """
    评估数据集中存在序列偏差和习题偏差问题的点
    :param seq_biased_point:
    :param statics_train:
    :param most_accuracy:
    :return:
    """
    # todo: 如果seq_biased_label全为0或者全为1，则不能计算AUC，这个还未处理
    double_biased_label = []
    double_biased_predict_score = []
    double_biased_predict_label = []
    for q_id, p_score, p_label in zip(seq_biased_point["low_acc_but_right"]["question"],
                                      seq_biased_point["low_acc_but_right"]["predict_score"],
                                      seq_biased_point["low_acc_but_right"]["predict_label"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        if q_acc_statics < 0:
            continue
        if q_acc_statics < most_accuracy:
            double_biased_label.append(1)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    for q_id, p_score, p_label in zip(seq_biased_point["high_acc_but_wrong"]["question"],
                                      seq_biased_point["high_acc_but_wrong"]["predict_score"],
                                      seq_biased_point["high_acc_but_wrong"]["predict_label"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        if q_acc_statics < 0:
            continue
        if q_acc_statics > (1 - most_accuracy):
            double_biased_label.append(0)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    return get_performance_no_error(double_biased_predict_score, double_biased_predict_label, double_biased_label)


def evaluate_core(predict_score, ground_truth, question_ids, allow_replace=True):
    question_ids_ = np.unique(question_ids)
    predict_score_balanced = []
    ground_truth_balanced = []

    for q_id in question_ids_:
        predict_score4q_id = predict_score[question_ids == q_id]
        ground_truth4q_id = ground_truth[question_ids == q_id]
        num_right = np.sum(ground_truth4q_id == 1)
        num_wrong = np.sum(ground_truth4q_id == 0)

        if num_right == 0 or num_wrong == 0:
            continue

        # 从label为1和0的测试数据中随机选相同数量（官方提供的代码上来看，是允许重复选取的）
        if allow_replace:
            num_balance = (num_wrong + num_right) // 2
        else:
            num_balance = min(num_wrong, num_right)
        index_right = np.random.choice(np.where(ground_truth4q_id == 1)[0], num_balance, replace=allow_replace)
        index_wrong = np.random.choice(np.where(ground_truth4q_id == 0)[0], num_balance, replace=allow_replace)
        index_balanced = list(index_right) + list(index_wrong)
        predict_score_balanced.append(predict_score4q_id[index_balanced])
        ground_truth_balanced.append(ground_truth4q_id[index_balanced])

    predict_score_balanced = np.concatenate(predict_score_balanced)
    ground_truth_balanced = np.concatenate(ground_truth_balanced)
    predict_label_balanced = [0 if p < 0.5 else 1 for p in predict_score_balanced]

    return get_performance_no_error(predict_score_balanced, predict_label_balanced, ground_truth_balanced)
