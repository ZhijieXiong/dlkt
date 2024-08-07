import numpy as np

from lib.util.parse import concept2question_from_Q

Q_table_path = "/Users/dream/myProjects/dlkt/lab/dataset_preprocessed/assist2012/Q_table_single_concept.npy"
target_concept = 22
Q_table = np.load(Q_table_path)
concept2question = concept2question_from_Q(Q_table)
print(concept2question[target_concept])
