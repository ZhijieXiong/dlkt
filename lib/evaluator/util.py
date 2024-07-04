import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error


def get_performance(item_list, item_pg_all):
    """
    获取指定item（习题或者知识点）上的性能
    :param item_list: 习题或者知识点的id list，即想要获取的那部分
    :param item_pg_all: 形式为{item1: [(p1, g1), (p2, g2), ...], item2: [(p1, g1), (p2, g2), ...], ...}
    :return:
    """
    predict_score = []
    ground_truth = []

    for item in item_list:
        p_list = list(map(lambda x: x[0], item_pg_all[item]))
        g_list = list(map(lambda x: x[1], item_pg_all[item]))
        predict_score += p_list
        ground_truth += g_list
    predict_label = [1 if p >= 0.5 else 0 for p in predict_score]

    return get_performance_no_error(predict_score, predict_label, ground_truth)


def get_performance_qc(question_list, concept_list, qc_pg_all):
    """
    获取指定习题以及知识点上的性能
    :param question_list:
    :param concept_list:
    :param qc_pg_all: 形式为{"q1_c1": [(p1, g1), (p2, g2), ...], "q2_c2": [(p1, g1), (p2, g2), ...], ...}
    :return:
    """
    predict_score = []
    ground_truth = []

    qc_all = []
    for q_id in question_list:
        for c_id in concept_list:
            qc_all.append(f"{q_id}_{c_id}")

    for qc_id in qc_all:
        p_list = list(map(lambda x: x[0], qc_pg_all[qc_id]))
        g_list = list(map(lambda x: x[1], qc_pg_all[qc_id]))
        predict_score += p_list
        ground_truth += g_list
    predict_label = [1 if p >= 0.5 else 0 for p in predict_score]

    return get_performance_no_error(predict_score, predict_label, ground_truth)


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
        return -1.0
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
    seq_fine_grained_sample = get_seq_fine_grained_sample(all_batch, window_seq_len, acc_th)
    num_easy = len(seq_fine_grained_sample["easy"]["question"])
    num_normal = len(seq_fine_grained_sample["normal"]["question"])
    num_hard = len(seq_fine_grained_sample["hard"]["question"])
    num_hard_label1 = sum(seq_fine_grained_sample["hard"]["ground_truth"])
    num_hard_label0 = num_hard - num_hard_label1

    return num_easy, num_normal, num_hard, num_hard_label0, num_hard_label1


def get_seq_fine_grained_performance(all_batch, window_seq_len, acc_th):
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


def get_num_question_fine_grained_sample(all_batch, statics_train, acc_th):
    seq_fine_grained_sample = get_question_fine_grained_sample(all_batch, statics_train, acc_th)
    num_easy = len(seq_fine_grained_sample["easy"]["question"])
    num_normal = len(seq_fine_grained_sample["normal"]["question"])
    num_hard = len(seq_fine_grained_sample["hard"]["question"])
    num_hard_label1 = sum(seq_fine_grained_sample["hard"]["ground_truth"])
    num_hard_label0 = num_hard - num_hard_label1

    return num_easy, num_normal, num_hard, num_hard_label0, num_hard_label1


def get_question_fine_grained_performance(all_batch, statics_train, acc_th):
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
    seq_easy_sample = seq_fine_grained_sample["easy"]
    seq_hard_sample = seq_fine_grained_sample["hard"]
    double_easy_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    double_hard_sample = {
        "question": [],
        "predict_score": [],
        "predict_label": [],
        "ground_truth": []
    }
    for i, q_id in enumerate(seq_easy_sample["question"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        ps = seq_easy_sample["predict_score"][i]
        pl = seq_easy_sample["predict_label"][i]
        gt = seq_easy_sample["ground_truth"][i]
        if (q_acc_statics > (1 - acc_th)) or (q_acc_statics < acc_th):
            double_easy_sample["question"].append(q_id)
            double_easy_sample["predict_score"].append(ps)
            double_easy_sample["predict_label"].append(pl)
            double_easy_sample["ground_truth"].append(gt)

    for i, q_id in enumerate(seq_hard_sample["question"]):
        q_acc_statics = statics_train["question_acc"][q_id]
        ps = seq_hard_sample["predict_score"][i]
        pl = seq_hard_sample["predict_label"][i]
        gt = seq_hard_sample["ground_truth"][i]
        if (q_acc_statics > (1 - acc_th)) or (q_acc_statics < acc_th):
            double_hard_sample["question"].append(q_id)
            double_hard_sample["predict_score"].append(ps)
            double_hard_sample["predict_label"].append(pl)
            double_hard_sample["ground_truth"].append(gt)

    double_fine_grained_sample = {
        "easy": double_easy_sample,
        "hard": double_hard_sample
    }

    return double_fine_grained_sample


def get_double_fine_grained_performance(all_batch, statics_train, window_seq_len, acc_th):
    double_fine_grained_sample = get_double_fine_grained_sample(all_batch, statics_train, window_seq_len, acc_th)
    easy_sample = double_fine_grained_sample["easy"]
    hard_sample = double_fine_grained_sample["hard"]

    performance_result = {
        "easy": get_performance_no_error(
            easy_sample["predict_score"], easy_sample["predict_label"], easy_sample["ground_truth"]
        ),
        "hard": get_performance_no_error(
            hard_sample["predict_score"], hard_sample["predict_label"], hard_sample["ground_truth"]
        ),
    }

    return performance_result


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
        x_normal = []
        y_normal = []
        x_hard = []
        y_hard = []
        x_input_unbalanced = []
        y_input_unbalanced = []
        for xx, yy, ll in zip(x, y, his_ave_record[window_len]["current_label"]):
            if his_acc_th <= xx <= (1 - his_acc_th):
                x_normal.append(xx)
                y_normal.append(yy)
            elif ((xx < his_acc_th) and (ll == 0)) or ((xx > (1 - his_acc_th)) and (ll == 1)):
                x_easy.append(xx)
                y_easy.append(yy)
            else:
                x_hard.append(xx)
                y_hard.append(yy)

            if not(his_acc_th <= xx <= (1 - his_acc_th)):
                x_input_unbalanced.append(xx)
                y_input_unbalanced.append(yy)

        result[window_len]["hard"] = get_ppmcc_no_error(x_hard, y_hard)
        result[window_len]["normal"] = get_ppmcc_no_error(x_normal, y_normal)
        result[window_len]["easy"] = get_ppmcc_no_error(x_easy, y_easy)
        result[window_len]["unbalanced"] = get_ppmcc_no_error(x_input_unbalanced, y_input_unbalanced)

    return result


def cal_PPMCC_train_question_acc_and_cur_model_pred(all_batch, statics_train, acc_th):
    """
    计算当前预测和习题在训练集中正确率的相关系数PPMCC\n
    :param all_batch:
    :param statics_train:
    :param acc_th:
    :return:
    """
    sample_record = {
        "train_q_acc": [],
        "current_predict_score": [],
        "current_label": []
    }
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["predict_score_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, predict_score_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq):
                if m == 0:
                    break
                q_id = question_seq[i]
                q_acc_statics = statics_train["question_acc"][q_id]

                if q_acc_statics != -1:
                    sample_record["train_q_acc"].append(q_acc_statics)
                    sample_record["current_predict_score"].append(predict_score_seq[i])
                    sample_record["current_label"].append(label_seq[i])

    result = {}
    x = sample_record["train_q_acc"]
    y = sample_record["current_predict_score"]
    result["all"] = get_ppmcc_no_error(x, y)

    x_easy = []
    y_easy = []
    x_normal = []
    y_normal = []
    x_hard = []
    y_hard = []
    x_input_unbalanced = []
    y_input_unbalanced = []
    for xx, yy, ll in zip(x, y, sample_record["current_label"]):
        if acc_th <= xx <= (1 - acc_th):
            x_normal.append(xx)
            y_normal.append(yy)
        elif ((xx < acc_th) and (ll == 0)) or ((xx > (1 - acc_th)) and (ll == 1)):
            x_easy.append(xx)
            y_easy.append(yy)
        else:
            x_hard.append(xx)
            y_hard.append(yy)

        if not (acc_th <= xx <= (1 - acc_th)):
            x_input_unbalanced.append(xx)
            y_input_unbalanced.append(yy)

    result["hard"] = get_ppmcc_no_error(x_hard, y_hard)
    result["normal"] = get_ppmcc_no_error(x_normal, y_normal)
    result["easy"] = get_ppmcc_no_error(x_easy, y_easy)
    result["unbalanced"] = get_ppmcc_no_error(x_input_unbalanced, y_input_unbalanced)

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
        x_normal = []
        y_normal = []
        x_hard = []
        y_hard = []
        x_input_unbalanced = []
        y_input_unbalanced = []
        for xx, yy, ll in zip(x, y, his_ave_record[window_len]["current_label"]):
            if his_acc_th <= xx <= (1 - his_acc_th):
                x_normal.append(xx)
                y_normal.append(yy)
            elif ((xx < his_acc_th) and (ll == 0)) or ((xx > (1 - his_acc_th)) and (ll == 1)):
                x_easy.append(xx)
                y_easy.append(yy)
            else:
                x_hard.append(xx)
                y_hard.append(yy)

            if not(his_acc_th <= xx <= (1 - his_acc_th)):
                x_input_unbalanced.append(xx)
                y_input_unbalanced.append(yy)

        result[window_len]["easy"] = get_ppmcc_no_error(x_easy, y_easy)
        result[window_len]["normal"] = get_ppmcc_no_error(x_normal, y_normal)
        result[window_len]["hard"] = get_ppmcc_no_error(x_hard, y_hard)
        result[window_len]["unbalanced"] = get_ppmcc_no_error(x_input_unbalanced, y_input_unbalanced)

    return result


def cal_PPMCC_train_question_acc_and_cur_label(all_batch, statics_train, acc_th):
    """
    计算当前标签和习题在训练集中正确率的相关系数PPMCC\n
    :param all_batch:
    :param statics_train:
    :param acc_th:
    :return:
    """
    sample_record = {
        "train_q_acc": [],
        "current_label": []
    }
    for batch in all_batch:
        zip_iter = zip(batch["question_seqs"], batch["label_seqs"], batch["mask_seqs"])
        for question_seq, label_seq, mask_seq in zip_iter:
            for i, m in enumerate(mask_seq):
                if m == 0:
                    break
                q_id = question_seq[i]
                q_acc_statics = statics_train["question_acc"][q_id]

                if q_acc_statics != -1:
                    sample_record["train_q_acc"].append(q_acc_statics)
                    sample_record["current_label"].append(label_seq[i])

    result = {}
    x = sample_record["train_q_acc"]
    y = sample_record["current_label"]
    result["all"] = get_ppmcc_no_error(x, y)

    x_easy = []
    y_easy = []
    x_normal = []
    y_normal = []
    x_hard = []
    y_hard = []
    x_input_unbalanced = []
    y_input_unbalanced = []
    for xx, yy, ll in zip(x, y, sample_record["current_label"]):
        if acc_th <= xx <= (1 - acc_th):
            x_normal.append(xx)
            y_normal.append(yy)
        elif ((xx < acc_th) and (ll == 0)) or ((xx > (1 - acc_th)) and (ll == 1)):
            x_easy.append(xx)
            y_easy.append(yy)
        else:
            x_hard.append(xx)
            y_hard.append(yy)

        if not (acc_th <= xx <= (1 - acc_th)):
            x_input_unbalanced.append(xx)
            y_input_unbalanced.append(yy)

    result["hard"] = get_ppmcc_no_error(x_hard, y_hard)
    result["normal"] = get_ppmcc_no_error(x_normal, y_normal)
    result["easy"] = get_ppmcc_no_error(x_easy, y_easy)
    result["unbalanced"] = get_ppmcc_no_error(x_input_unbalanced, y_input_unbalanced)

    return result
