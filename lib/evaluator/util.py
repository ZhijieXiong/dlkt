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


def evaluate4seq_len(label_dis, score_dis, split_len, split_percent=None):
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
    print("按照序列长度划分，例如(0, 50)表示长度在这区间的序列")
    for i in range(len(label_dis4len)):
        if len(label_dis4len[i]) == 0:
            continue
        g = np.array(label_dis4len[i])
        p = np.array(score_dis4len[i])
        p_label = [1 if _ >= 0.5 else 0 for _ in p]
        answer_acc = g.sum() / len(g)
        print(f"({indices4len[i][0]:<3}, {indices4len[i][1]:<3}), num of samples is {g.size:<10}, "
              f"acc of answer is {answer_acc*100:<9.3}%: "
              f"auc is {metrics.roc_auc_score(y_true=g, y_score=p):<9.5}, "
              f"acc is {metrics.accuracy_score(g, p_label):<9.5}")

    # if split_percent is None:
    #     split_percent = [0, 0.25, 0.5, 0.75, 1]
    # label_dis4percent = [[] for _ in range(len(split_percent) - 1)]
    # score_dis4percent = [[] for _ in range(len(split_percent) - 1)]
    # label_dis4len_every = [[] for _ in range(len(split_len) - 1)]
    # score_dis4len_every = [[] for _ in range(len(split_len) - 1)]
    # max_seq_len = len(label_dis)
    # if max_seq_len < 20:
    #     return
    # indices4percent = []
    # for i in range(len(split_percent)):
    #     if i == len(split_percent) - 1:
    #         break
    #     indices4percent.append((split_percent[i], split_percent[i+1]))
    # # 长度小于50的不用测
    # for seq_len in range(48, len(label_dis)):
    #     # 实际序列长度是seq_len + 1
    #     current_indices = []
    #     for percent_tuple in indices4percent:
    #         current_indices.append((int((seq_len + 1) * percent_tuple[0]), int((seq_len + 1) * percent_tuple[1])))
    #     tmp_label_dis8seq_len = label_dis[seq_len]
    #     tmp_score_dis8seq_len = score_dis[seq_len]
    #     if len(tmp_label_dis8seq_len) == 0:
    #         continue
    #     label_dis8seq_len = np.array(tmp_label_dis8seq_len).reshape((-1, seq_len+1))
    #     score_dis8seq_len = np.array(tmp_score_dis8seq_len).reshape((-1, seq_len+1))
    #     for i, indices in enumerate(current_indices):
    #         index_s, index_e = indices
    #         tmp_label_dis = label_dis8seq_len[::, index_s: index_e].tolist()
    #         tmp_score_dis = score_dis8seq_len[::, index_s: index_e].tolist()
    #         label_dis4percent[i] += [item for item_list in tmp_label_dis for item in item_list]
    #         score_dis4percent[i] += [item for item_list in tmp_score_dis for item in item_list]
    # print("按照序列长度百分比划分，例如(0, 0.2)表示取每段序列（长度大于等于50）的(0, 0.2)部分")
    # for i in range(len(label_dis4percent)):
    #     if len(label_dis4percent[i]) == 0:
    #         continue
    #     g = np.array(label_dis4percent[i])
    #     p = np.array(score_dis4percent[i])
    #     p_label = [1 if _ >= 0.5 else 0 for _ in p]
    #     print(f"({split_percent[i]}, {split_percent[i+1]}), num of samples is {g.size}: "
    #           f"auc is {metrics.roc_auc_score(y_true=g, y_score=p)}, acc is {metrics.accuracy_score(g, p_label)}")
    #
    # for seq_len in range(len(label_dis)):
    #     k = 0
    #     for indices in indices4len:
    #         if indices[0] <= seq_len < indices[1]:
    #             break
    #         k += 1
    #     index_s = indices4len[k][0]
    #     tmp_label_dis8seq_len = label_dis[seq_len]
    #     tmp_score_dis8seq_len = score_dis[seq_len]
    #     if len(tmp_label_dis8seq_len) == 0:
    #         continue
    #     label_dis8seq_len = np.array(tmp_label_dis8seq_len).reshape((-1, seq_len + 1))
    #     score_dis8seq_len = np.array(tmp_score_dis8seq_len).reshape((-1, seq_len + 1))
    #     tmp_label_dis = label_dis8seq_len[::, index_s::].tolist()
    #     tmp_score_dis = score_dis8seq_len[::, index_s::].tolist()
    #     label_dis4len_every[k] += [item for item_list in tmp_label_dis for item in item_list]
    #     score_dis4len_every[k] += [item for item_list in tmp_score_dis for item in item_list]
    # print("按照序列长度划分，例如(0, 50)表示取每段序列的(0, 50)部分")
    # for i in range(len(label_dis4len_every)):
    #     if len(label_dis4len_every[i]) == 0:
    #         continue
    #     g = np.array(label_dis4len_every[i])
    #     p = np.array(score_dis4len_every[i])
    #     p_label = [1 if _ >= 0.5 else 0 for _ in p]
    #     print(f"{indices4len[i]}, num of samples is {g.size}: : "
    #           f"auc is {metrics.roc_auc_score(y_true=g, y_score=p)}, acc is {metrics.accuracy_score(g, p_label)}")
