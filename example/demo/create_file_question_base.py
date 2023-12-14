import numpy as np

from lib.util.data import write2file, read_preprocessed_file
from lib.dataset.KTDataset import KTDataset

Q_table = np.load(r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\assist2009\Q_table_multi_concept.npy")
fpath = r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2009_test_split_8.txt"
data = read_preprocessed_file(fpath)

write2file(
    KTDataset.dataset_multi_concept2question_pykt(data, Q_table, 3, 200, 4),
    fpath.replace(".txt", "_question_base4multi_concept.txt")
)
