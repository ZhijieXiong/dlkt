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


def get_ppmcc_no_error(x, y):
    assert len(x) == len(y), f"length of x and y must be equal"
    if len(x) == 0:
        return -1
    return np.corrcoef(x, y)[0, 1]


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


def get_seq_fine_grained_sample(all_batch, previous_seq_len, seq_most_accuracy):
    """
    :param all_batch:
    :param previous_seq_len:
    :param seq_most_accuracy:
    :return:
    """
    easy_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    normal_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    hard_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    cold_start_sample = {
        "question": [],
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
                cold_start_sample["question"].append(question_seq[i])
                cold_start_sample["predict_score"].append(predict_score_seq[i])
                cold_start_sample["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                cold_start_sample["ground_truth"].append(label_seq[i])

            for i, m in enumerate(mask_seq[previous_seq_len:]):
                j = i + previous_seq_len
                if m == 0:
                    break

                context_label = label_seq[j-previous_seq_len:j]
                context_accuracy = sum(context_label) / len(context_label)

                if seq_most_accuracy <= context_accuracy <= (1 - seq_most_accuracy):
                    normal_sample["question"].append(question_seq[j])
                    normal_sample["predict_score"].append(predict_score_seq[j])
                    normal_sample["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    normal_sample["ground_truth"].append(label_seq[j])
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 0)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 1)):
                    hard_sample["question"].append(question_seq[j])
                    hard_sample["predict_score"].append(predict_score_seq[j])
                    hard_sample["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    hard_sample["ground_truth"].append(label_seq[j])
                elif ((context_accuracy > (1 - seq_most_accuracy)) and (label_seq[j] == 1)) or \
                        ((context_accuracy < seq_most_accuracy) and (label_seq[j] == 0)):
                    easy_sample["question"].append(question_seq[j])
                    easy_sample["predict_score"].append(predict_score_seq[j])
                    easy_sample["predict_label"].append(1 if (predict_score_seq[j] > 0.5) else 0)
                    easy_sample["ground_truth"].append(label_seq[j])

    fine_grained_sample = {
        "easy": easy_sample,
        "normal": normal_sample,
        "hard": hard_sample,
        "cold_start": cold_start_sample

    }

    return fine_grained_sample


def get_num_seq_fine_grained_sample(all_batch, window_seq_len, acc_th):
    """
    :param all_batch:
    :param window_seq_len:
    :param acc_th:
    :return:
    """
    seq_fine_grained_sample = get_seq_fine_grained_sample(all_batch, window_seq_len, acc_th)
    num_easy = len(seq_fine_grained_sample["easy"]["question"])
    num_normal = len(seq_fine_grained_sample["normal"]["question"])
    num_hard = len(seq_fine_grained_sample["hard"]["question"])
    num_cold_start = len(seq_fine_grained_sample["cold_start"]["question"])
    num_warm_started = num_easy + num_normal + num_hard

    return num_easy, num_normal, num_hard, num_cold_start, num_warm_started


def get_seq_fine_grained_performance(all_batch, window_seq_len, acc_th):
    """
    :param all_batch:
    :param window_seq_len:
    :param acc_th:
    :return:
    """
    seq_fine_grained_sample = get_seq_fine_grained_sample(all_batch, window_seq_len, acc_th)
    easy_sample = seq_fine_grained_sample["easy"]
    normal_sample = seq_fine_grained_sample["normal"]
    hard_sample = seq_fine_grained_sample["hard"]
    cold_start_sample = seq_fine_grained_sample["cold_start"]

    performance_result = {
        "easy": get_performance_no_error(
            easy_sample["predict_score"], easy_sample["predict_label"], easy_sample["ground_truth"]
        ),
        "normal": get_performance_no_error(
            normal_sample["predict_score"], normal_sample["predict_label"], normal_sample["ground_truth"]
        ),
        "hard": get_performance_no_error(
            hard_sample["predict_score"], hard_sample["predict_label"], hard_sample["ground_truth"]
        ),
        "cold_start": get_performance_no_error(
            cold_start_sample["predict_score"], cold_start_sample["predict_label"], cold_start_sample["ground_truth"]
        )
    }

    return performance_result


def get_question_fine_grained_sample(all_batch, statics_train, most_accuracy):
    """
    返回满足以下条件的点：该习题是高正确率习题，但是做错，或者，该习题是低正确率习题，但是做对\n
    :param all_batch:
    :param statics_train:
    :param most_accuracy:
    :return:
    """
    unseen_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    easy_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    normal_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    hard_sample = {
        "question": [],
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
                if q_acc_statics < 0:
                    unseen_sample["question"].append(question_seq[i])
                    unseen_sample["predict_score"].append(predict_score_seq[i])
                    unseen_sample["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    unseen_sample["ground_truth"].append(label_seq[i])
                elif ((q_acc_statics > (1 - most_accuracy)) and (label == 1)) or \
                        ((q_acc_statics < most_accuracy) and (label == 0)):
                    easy_sample["question"].append(question_seq[i])
                    easy_sample["predict_score"].append(predict_score_seq[i])
                    easy_sample["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    easy_sample["ground_truth"].append(label_seq[i])
                elif most_accuracy <= q_acc_statics <= (1 - most_accuracy):
                    normal_sample["question"].append(question_seq[i])
                    normal_sample["predict_score"].append(predict_score_seq[i])
                    normal_sample["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    normal_sample["ground_truth"].append(label_seq[i])
                elif ((q_acc_statics > (1 - most_accuracy)) and (label == 0)) or \
                        ((q_acc_statics < most_accuracy) and (label == 1)):
                    hard_sample["question"].append(question_seq[i])
                    hard_sample["predict_score"].append(predict_score_seq[i])
                    hard_sample["predict_label"].append(1 if (predict_score_seq[i] > 0.5) else 0)
                    hard_sample["ground_truth"].append(label_seq[i])

    fine_grained_sample = {
        "easy": easy_sample,
        "normal": normal_sample,
        "hard": hard_sample,
        "unseen": unseen_sample
    }

    return fine_grained_sample


def get_question_fine_grained_performance(all_batch, statics_train, acc_th):
    """
    返回满足以下条件的点：该习题是高正确率习题，但是做错，或者，该习题是低正确率习题，但是做对\n
    :param all_batch:
    :param statics_train:
    :param acc_th:
    :return:
    """
    question_fine_grained_sample = get_question_fine_grained_sample(all_batch, statics_train, acc_th)
    easy_sample = question_fine_grained_sample["easy"]
    normal_sample = question_fine_grained_sample["normal"]
    hard_sample = question_fine_grained_sample["hard"]
    unseen_sample = question_fine_grained_sample["unseen"]

    performance_result = {
        "easy": get_performance_no_error(
            easy_sample["predict_score"], easy_sample["predict_label"], easy_sample["ground_truth"]
        ),
        "normal": get_performance_no_error(
            normal_sample["predict_score"], normal_sample["predict_label"], normal_sample["ground_truth"]
        ),
        "hard": get_performance_no_error(
            hard_sample["predict_score"], hard_sample["predict_label"], hard_sample["ground_truth"]
        ),
        "unseen": get_performance_no_error(
            unseen_sample["predict_score"], unseen_sample["predict_label"], unseen_sample["ground_truth"]
        )
    }

    return performance_result


def get_double_fine_grained_sample(all_batch, statics_train, window_seq_len, acc_th):
    seq_fine_grained_sample = get_seq_fine_grained_sample(all_batch, window_seq_len, acc_th)
    double_easy_sample = {

    }
    double_hard_sample = {

    }
    pass


def cal_PPMCC_his_acc_and_cur_model_pred(all_batch, window_lens, his_acc_th):
    """
    计算当前预测和历史（一定窗口长度）正确率的相关系数PPMCC\n
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
        result[window_len]["all"] = get_ppmcc_no_error(x, y)

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

        result[window_len]["hard"] = get_ppmcc_no_error(x_hard, y_hard)
        result[window_len]["easy"] = get_ppmcc_no_error(x_easy, y_easy)

    return result


def cal_PPMCC_his_acc_and_cur_label(all_batch, window_lens, his_acc_th):
    """
    计算当前标签和历史（一定窗口长度）正确率的相关系数PPMCC\n
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
        result[window_len]["all"] = get_ppmcc_no_error(x, y)

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

        result[window_len]["easy"] = get_ppmcc_no_error(x_easy, y_easy)
        result[window_len]["hard"] = get_ppmcc_no_error(x_hard, y_hard)

    return result
