import numpy as np
from collections import defaultdict

from .parse import question2concept_from_Q


def RCD_construct_dependency_matrix(data_uniformed, num_concept, Q_table):
    """
    RCD所使用的构造知识点关联矩阵的方法，要求输入的data_uniformed为single concept或者是only question
    :param data_uniformed:
    :param num_concept:
    :param Q_table:
    :return:
    """
    question2concept = question2concept_from_Q(Q_table)
    edge_dic_deno = {}

    # Calculate correct matrix
    concept_correct = np.zeros([num_concept, num_concept])
    for item_data in data_uniformed:
        if item_data["seq_len"] < 3:
            continue

        for log_i in range(item_data["seq_len"] - 1):
            if item_data["correct_seq"][log_i] * item_data["correct_seq"][log_i+1] == 1:
                current_cs = question2concept[item_data["question_seq"][log_i]]
                next_cs = question2concept[item_data["question_seq"][log_i + 1]]
                for ci in current_cs:
                    for cj in next_cs:
                        if ci != cj:
                            concept_correct[ci][cj] += 1.0
                            # calculate the number of correctly answering i
                            edge_dic_deno.setdefault(ci, 1)
                            edge_dic_deno[ci] += 1

    s = 0
    c = 0
    # Calculate transition matrix
    concept_directed = np.zeros([num_concept, num_concept])
    for i in range(num_concept):
        for j in range(num_concept):
            if i != j and concept_correct[i][j] > 0:
                concept_directed[i][j] = float(concept_correct[i][j]) / edge_dic_deno[i]
                s += concept_directed[i][j]
                c += 1
    o = np.zeros([num_concept, num_concept])
    min_c = 100000
    max_c = 0
    for i in range(num_concept):
        for j in range(num_concept):
            if concept_correct[i][j] > 0 and i != j:
                min_c = min(min_c, concept_directed[i][j])
                max_c = max(max_c, concept_directed[i][j])
    s_o = 0
    l_o = 0
    for i in range(num_concept):
        for j in range(num_concept):
            if concept_correct[i][j] > 0 and i != j:
                o[i][j] = (concept_directed[i][j] - min_c) / (max_c - min_c)
                l_o += 1
                s_o += o[i][j]

    # avg^2 is threshold
    threshold1 = s_o / l_o
    threshold1 *= threshold1

    # 8/2 threshold
    k = int(num_concept * num_concept * 0.1)
    o_1d = o.reshape(-1)
    threshold2 = min(o_1d[np.argpartition(o_1d, -k)[-k:]])

    threshold = max(threshold1, threshold2)

    edge = np.zeros([num_concept, num_concept])
    for i in range(num_concept):
        for j in range(num_concept):
            if o[i][j] >= threshold:
                edge[i][j] = 1

    return edge


def RCD_process_edge(edge):
    concept_undirected = np.minimum(edge, edge.T)
    concept_directed = np.maximum(edge - edge.T, 0)
    return concept_undirected, concept_directed


def undirected_graph2similar(G):
    # G: n * n, G[i,j]==1 --> i is similar with j
    node_dict = {}
    for i in range(len(G)):
        node_dict[i] = [i]
        for j in range(len(G)):
            if i != j and G[i][j] == 1:
                node_dict[i].append(j)

    return node_dict


def directed_graph2pre_and_post(G):
    pre_dict = defaultdict(list)
    post_dict = defaultdict(list)
    for i in range(len(G)):
        for j in range(len(G)):
            if i != j and G[i][j] == 1:
                pre_dict[j].append(i)
                post_dict[i].append(j)

    node_dict = {}
    for i in range(len(G)):
        node_dict[i] = {
            "pre": pre_dict[i],
            "post": post_dict[i]
        }

    return node_dict
