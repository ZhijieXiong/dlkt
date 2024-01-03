import numpy as np

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
    avg = s_o / l_o  # total / count
    # avg is threshold
    graph = ''
    edge = np.zeros([num_concept, num_concept])
    for i in range(num_concept):
        for j in range(num_concept):
            if o[i][j] >= avg:
                graph += str(i) + '\t' + str(j) + '\n'
                edge[i][j] = 1
    # e_l = []
    # co = 0
    # tr = 0
    # all_r = 0
    # # Calculate concept dependency relation
    # for i in range(num_concept):
    #     for j in range(num_concept):
    #         if (i, j) not in e_l:
    #             e_l.append((i, j))
    #             if edge[i][j] == 1:
    #                 if edge[j][i] == 1:
    #                     co += 1
    #                     all_r += 1
    #                     e_l.append((j, i))
    #                 else:
    #                     tr += 1
    #                     all_r += 1

    return edge


def RCD_process_edge(edge, num_concept):
    concept_directed = np.zeros([num_concept, num_concept])
    concept_undirected = np.zeros([num_concept, num_concept])
    visit = []
    for i in range(num_concept):
        for j in range(num_concept):
            if edge[i][j] != 0:
                e = (i, j)
                if e not in visit:
                    if edge[j][i] != 0:
                        concept_undirected[i][j] = 1
                        visit.append(e)
                        visit.append((e[1], e[0]))
                    else:
                        concept_directed[i][j] = 1
                        visit.append(e)
    return concept_undirected, concept_directed
