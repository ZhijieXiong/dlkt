import numpy as np


def concept2question_from_Q(Q_table):
    # 将Q table转换为{concept_id1: [question_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[:, i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[1])}
    return result


def question2concept_from_Q(Q_table):
    # 将Q table转换为{question_id1: [concept_id1,...]}的形式，表示各个知识点对应的习题
    result = {i: np.argwhere(Q_table[i] == 1).reshape(-1).tolist() for i in range(Q_table.shape[0])}
    return result
