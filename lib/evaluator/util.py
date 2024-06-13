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
        "RMSE": mean_squared_error(y_true=true_label, y_pred=predict_score) ** 0.5,
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
                j = i + previous_seq_len
                if m == 0:
                    break

                context_labels = label_seq[j-previous_seq_len:j]
                context_accuracy = sum(context_labels) / len(context_labels)

                if (context_accuracy < seq_most_accuracy) and (label_seq[j] == 1):
                    result["low_acc_but_right"]["question"].append(question_seq[j])
                    result["low_acc_but_right"]["predict_score"].append(predict_score_seq[j])
                    result["low_acc_but_right"]["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                elif (context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 0):
                    result["high_acc_but_wrong"]["question"].append(question_seq[j])
                    result["high_acc_but_wrong"]["predict_score"].append(predict_score_seq[j])
                    result["high_acc_but_wrong"]["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
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


def get_seq_easy_point(all_batch, previous_seq_len, seq_most_accuracy):
    """
    返回每条序列中满足以下条件的点：\n
    1. 该点的context seq len为previous_seq_len，即从每条序列的第previous_seq_len个点开始\n
    2. 该点的context seq正确率大于(1 - seq_most_accuracy)，且做对，或者小于seq_most_accuracy，且做错\n
    :param all_batch:
    :param previous_seq_len:
    :param seq_most_accuracy:
    :return:
    """
    result = {
        "high_acc_and_right": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        },
        "low_acc_and_wrong": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        }
    }
    non_seq_easy = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["predict_score_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, predict_score_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq[:previous_seq_len]):
                if m == 0:
                    break
                non_seq_easy["predict_score"].append(predict_score_seq[i])
                non_seq_easy["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                non_seq_easy["ground_truth"].append(label_seq[i])

            for i, m in enumerate(mask_seq[previous_seq_len:]):
                j = i + previous_seq_len
                if m == 0:
                    break

                context_label = label_seq[j-previous_seq_len:j]
                context_accuracy = sum(context_label) / len(context_label)

                if (context_accuracy < seq_most_accuracy) and (label_seq[j] == 0):
                    result["low_acc_and_wrong"]["question"].append(question_seq[j])
                    result["low_acc_and_wrong"]["predict_score"].append(predict_score_seq[j])
                    result["low_acc_and_wrong"]["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                elif (context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 1):
                    result["high_acc_and_right"]["question"].append(question_seq[j])
                    result["high_acc_and_right"]["predict_score"].append(predict_score_seq[j])
                    result["high_acc_and_right"]["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                else:
                    non_seq_easy["predict_score"].append(predict_score_seq[j])
                    non_seq_easy["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    non_seq_easy["ground_truth"].append(label_seq[j])

    return result, non_seq_easy


def get_num_seq_fine_grained_point(all_batch, previous_seq_len, seq_most_accuracy):
    """
    :param all_batch:
    :param previous_seq_len:
    :param seq_most_accuracy:
    :return:
    """
    num_easy = 0
    num_normal = 0
    num_hard = 0
    num_cold_start = 0
    num_warm_started = 0
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq[:previous_seq_len]):
                if m == 0:
                    break
                num_cold_start += 1

            for i, m in enumerate(mask_seq[previous_seq_len:]):
                j = i + previous_seq_len
                if m == 0:
                    break

                num_warm_started += 1

                context_label = label_seq[j-previous_seq_len:j]
                context_accuracy = sum(context_label) / len(context_label)

                if seq_most_accuracy <= context_accuracy <= (1 - seq_most_accuracy):
                    num_normal += 1
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 0)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 1)):
                    num_hard += 1
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 1)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 0)):
                    num_easy += 1

    return num_easy, num_normal, num_hard, num_cold_start, num_warm_started


def get_seq_fine_grained_performance(all_batch, previous_seq_len, seq_most_accuracy):
    """
    :param all_batch:
    :param previous_seq_len:
    :param seq_most_accuracy:
    :return:
    """
    seq_easy = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    seq_normal = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    seq_hard = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    cold_start = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    cold_start_and_normal = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    seq_normal_and_hard = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    cold_start_and_normal_and_hard = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    warm_started = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["predict_score_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, predict_score_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq[:previous_seq_len]):
                if m == 0:
                    break
                cold_start["predict_score"].append(predict_score_seq[i])
                cold_start["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                cold_start["ground_truth"].append(label_seq[i])
                cold_start_and_normal["predict_score"].append(predict_score_seq[i])
                cold_start_and_normal["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                cold_start_and_normal["ground_truth"].append(label_seq[i])
                cold_start_and_normal_and_hard["predict_score"].append(predict_score_seq[i])
                cold_start_and_normal_and_hard["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                cold_start_and_normal_and_hard["ground_truth"].append(label_seq[i])

            for i, m in enumerate(mask_seq[previous_seq_len:]):
                j = i + previous_seq_len
                if m == 0:
                    break

                warm_started["predict_score"].append(predict_score_seq[j])
                warm_started["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                warm_started["ground_truth"].append(label_seq[j])

                context_label = label_seq[j-previous_seq_len:j]
                context_accuracy = sum(context_label) / len(context_label)

                if seq_most_accuracy <= context_accuracy <= (1 - seq_most_accuracy):
                    seq_normal["predict_score"].append(predict_score_seq[j])
                    seq_normal["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    seq_normal["ground_truth"].append(label_seq[j])
                    seq_normal_and_hard["predict_score"].append(predict_score_seq[j])
                    seq_normal_and_hard["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    seq_normal_and_hard["ground_truth"].append(label_seq[j])
                    cold_start_and_normal["predict_score"].append(predict_score_seq[j])
                    cold_start_and_normal["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    cold_start_and_normal["ground_truth"].append(label_seq[j])
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 0)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 1)):
                    seq_hard["predict_score"].append(predict_score_seq[j])
                    seq_hard["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    seq_hard["ground_truth"].append(label_seq[j])
                    seq_normal_and_hard["predict_score"].append(predict_score_seq[j])
                    seq_normal_and_hard["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    seq_normal_and_hard["ground_truth"].append(label_seq[j])
                    cold_start_and_normal_and_hard["predict_score"].append(predict_score_seq[j])
                    cold_start_and_normal_and_hard["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    cold_start_and_normal_and_hard["ground_truth"].append(label_seq[j])
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 1)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 0)):
                    seq_easy["predict_score"].append(predict_score_seq[j])
                    seq_easy["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    seq_easy["ground_truth"].append(label_seq[j])

    result = {
        "easy": get_performance_no_error(
            seq_easy["predict_score"], seq_easy["predict_label"], seq_easy["ground_truth"]
        ),
        "normal": get_performance_no_error(
            seq_normal["predict_score"], seq_normal["predict_label"], seq_normal["ground_truth"]
        ),
        "hard": get_performance_no_error(
            seq_hard["predict_score"], seq_hard["predict_label"], seq_hard["ground_truth"]
        ),
        "cold_start": get_performance_no_error(
            cold_start["predict_score"], cold_start["predict_label"], cold_start["ground_truth"]
        ),
        "cold_start_and_normal": get_performance_no_error(
            cold_start_and_normal["predict_score"],
            cold_start_and_normal["predict_label"],
            cold_start_and_normal["ground_truth"]
        ),
        "normal_and_hard": get_performance_no_error(
            seq_normal_and_hard["predict_score"],
            seq_normal_and_hard["predict_label"],
            seq_normal_and_hard["ground_truth"]
        ),
        "cold_start_and_normal_and_hard": get_performance_no_error(
            cold_start_and_normal_and_hard["predict_score"],
            cold_start_and_normal_and_hard["predict_label"],
            cold_start_and_normal_and_hard["ground_truth"]
        ),
        "warm_started": get_performance_no_error(
            warm_started["predict_score"], warm_started["predict_label"], warm_started["ground_truth"]
        ),
    }

    return result


def get_question_easy_point(all_batch, statics_train, most_accuracy):
    """
    返回满足以下条件的点：该习题是高正确率习题，且做对，或者，该习题是低正确率习题，且做错\n
    :param all_batch:
    :param statics_train:
    :param most_accuracy:
    :return:
    """
    result = {
        "high_acc_and_right": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        },
        "low_acc_and_wrong": {
            "question": [],
            "predict_score": [],
            "predict_label": []
        }
    }
    non_question_easy = {
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
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
                if (q_acc_statics >= 0) and (q_acc_statics > (1 - most_accuracy)) and (label == 1):
                    result["high_acc_and_right"]["question"].append(q_id)
                    result["high_acc_and_right"]["predict_score"].append(predict_score_seq[i])
                    result["high_acc_and_right"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    continue
                if (q_acc_statics >= 0) and (q_acc_statics < most_accuracy) and (label == 0):
                    result["low_acc_and_wrong"]["question"].append(q_id)
                    result["low_acc_and_wrong"]["predict_score"].append(predict_score_seq[i])
                    result["low_acc_and_wrong"]["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    continue
                non_question_easy["predict_score"].append(predict_score_seq[i])
                non_question_easy["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                non_question_easy["ground_truth"].append(label)

    return result, non_question_easy


def evaluate_easy(seq_easy_point):
    seq_biased_label = [0] * len(seq_easy_point["low_acc_and_wrong"]["predict_score"]) + \
                       [1] * len(seq_easy_point["high_acc_and_right"]["predict_score"])
    seq_biased_predict_score = seq_easy_point["low_acc_and_wrong"]["predict_score"] + \
                               seq_easy_point["high_acc_and_right"]["predict_score"]
    seq_biased_predict_label = seq_easy_point["low_acc_and_wrong"]["predict_label"] + \
                               seq_easy_point["high_acc_and_right"]["predict_label"]

    return get_performance_no_error(seq_biased_predict_score, seq_biased_predict_label, seq_biased_label)


def evaluate_double_easy(seq_easy_point, statics_train, most_accuracy):
    double_biased_label = []
    double_biased_predict_score = []
    double_biased_predict_label = []
    for q_id, p_score, p_label in zip(seq_easy_point["low_acc_and_wrong"]["question"],
                                      seq_easy_point["low_acc_and_wrong"]["predict_score"],
                                      seq_easy_point["low_acc_and_wrong"]["predict_label"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        if q_acc_statics < 0:
            continue
        if q_acc_statics < most_accuracy:
            double_biased_label.append(0)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    for q_id, p_score, p_label in zip(seq_easy_point["high_acc_and_right"]["question"],
                                      seq_easy_point["high_acc_and_right"]["predict_score"],
                                      seq_easy_point["high_acc_and_right"]["predict_label"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        if q_acc_statics < 0:
            continue
        if q_acc_statics > (1 - most_accuracy):
            double_biased_label.append(1)
            double_biased_predict_score.append(p_score)
            double_biased_predict_label.append(p_label)

    return get_performance_no_error(double_biased_predict_score, double_biased_predict_label, double_biased_label)


def cal_ppmcc_no_error(x, y):
    assert len(x) == len(y), f"length of x and y must be equal"
    if len(x) == 0:
        return -1
    return np.corrcoef(x, y)[0, 1]


def cal_PPMCC_his_acc_and_cur_model_pred(all_batch, window_lens, his_acc_th):
    """
    计算当前预测和历史（一定窗口长度）正确率的相关系数PPMCC

    :param all_batch:
    :param window_lens:
    :param his_acc_th:
    :return:
    """
    his_ave_record = {}
    for window_len in window_lens:
        his_ave_record[window_len] = {
            "history_average_accuracy": [],
            "current_predict_score": [],
            "current_label": []
        }

        for batch in all_batch:
            zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["predict_score_seqs"], batch["mask_seqs"])
            for question_seq, label_seq, predict_score_seq, mask_seq in zip_iter:
                for i, m in enumerate(mask_seq[window_len:]):
                    i += window_len
                    if m == 0:
                        break

                    context_labels = label_seq[i - window_len:i]
                    context_accuracy = sum(context_labels) / len(context_labels)

                    his_ave_record[window_len]["history_average_accuracy"].append(context_accuracy)
                    his_ave_record[window_len]["current_predict_score"].append(predict_score_seq[i])
                    his_ave_record[window_len]["current_label"].append(label_seq[i])

    result = {}
    for window_len in window_lens:
        result[window_len] = {}
        # 不过滤，直接计算所有预测和历史的相关系数
        x = his_ave_record[window_len]["history_average_accuracy"]
        y = his_ave_record[window_len]["current_predict_score"]
        result[window_len]["all"] = cal_ppmcc_no_error(x, y)

        x_easy = []
        y_easy = []
        x_hard = []
        y_hard = []
        for xx, yy, ll in zip(x, y, his_ave_record[window_len]["current_label"]):
            if his_acc_th <= xx <= (1 - his_acc_th):
                x_hard.append(xx)
                y_hard.append(yy)
            else:
                x_easy.append(xx)
                y_easy.append(yy)

        result[window_len]["hard"] = cal_ppmcc_no_error(x_hard, y_hard)
        result[window_len]["easy"] = cal_ppmcc_no_error(x_easy, y_easy)

    return result


def cal_PPMCC_his_acc_and_cur_label(all_batch, window_lens, his_acc_th):
    """
    计算当前标签和历史（一定窗口长度）正确率的相关系数PPMCC

    :param all_batch:
    :param window_lens:
    :param his_acc_th:
    :return:
    """
    his_ave_record = {}
    for window_len in window_lens:
        his_ave_record[window_len] = {
            "history_average_accuracy": [],
            "current_label": []
        }

        for batch in all_batch:
            zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["mask_seqs"])
            for question_seq, label_seq, mask_seq in zip_iter:
                for i, m in enumerate(mask_seq[window_len:]):
                    i += window_len
                    if m == 0:
                        break

                    context_labels = label_seq[i - window_len:i]
                    context_accuracy = sum(context_labels) / len(context_labels)

                    his_ave_record[window_len]["history_average_accuracy"].append(context_accuracy)
                    his_ave_record[window_len]["current_label"].append(label_seq[i])

    result = {}
    for window_len in window_lens:
        result[window_len] = {}
        # 不过滤，直接计算所有标签和历史的相关系数
        x = his_ave_record[window_len]["history_average_accuracy"]
        y = his_ave_record[window_len]["current_label"]
        result[window_len]["all"] = cal_ppmcc_no_error(x, y)

        x_easy = []
        y_easy = []
        x_hard = []
        y_hard = []
        for xx, yy in zip(x, y):
            if his_acc_th <= xx <= (1 - his_acc_th):
                x_hard.append(xx)
                y_hard.append(yy)
            else:
                x_easy.append(xx)
                y_easy.append(yy)

        result[window_len]["easy"] = cal_ppmcc_no_error(x_easy, y_easy)
        result[window_len]["hard"] = cal_ppmcc_no_error(x_hard, y_hard)

    return result


def cal_PPMCC_his_acc_and_cur_label_new(all_batch, window_lens, question2concept):
    """
    计算当前标签和历史（一定窗口长度）正确率的相关系数PPMCC\n
    easy是历史窗口包含和当前习题相关练习记录，反之为hard

    :param all_batch:
    :param window_lens:
    :param question2concept:
    :return:
    """
    his_ave_record = {}
    for window_len in window_lens:
        his_ave_record[window_len] = {
            "history_average_accuracy": [],
            "current_label": [],
            "concept_did": []
        }

        for batch in all_batch:
            zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["mask_seqs"])
            for question_seq, label_seq, mask_seq in zip_iter:
                concept_context = list(map(lambda q: question2concept[q], question_seq))
                for i, m in enumerate(mask_seq[window_len:]):
                    i += window_len
                    if m == 0:
                        break

                    context_labels = label_seq[i - window_len:i]
                    context_accuracy = sum(context_labels) / len(context_labels)

                    context_concepts = set([c_id for c_ids in concept_context[i - window_len:i] for c_id in c_ids])
                    current_concept = set([c_id for c_id in concept_context[i]])

                    his_ave_record[window_len]["history_average_accuracy"].append(context_accuracy)
                    his_ave_record[window_len]["current_label"].append(label_seq[i])
                    his_ave_record[window_len]["concept_did"].append(bool(context_concepts & current_concept))

    result = {}
    for window_len in window_lens:
        result[window_len] = {}
        # 不过滤，直接计算所有标签和历史的相关系数
        x = his_ave_record[window_len]["history_average_accuracy"]
        y = his_ave_record[window_len]["current_label"]
        result[window_len]["all"] = cal_ppmcc_no_error(x, y)

        x_hard = []
        y_hard = []
        for xx, yy, concept_did in zip(x, y, his_ave_record[window_len]["concept_did"]):
            if not concept_did:
                x_hard.append(xx)
                y_hard.append(yy)

        result[window_len]["hard"] = cal_ppmcc_no_error(x_hard, y_hard)

    return result
