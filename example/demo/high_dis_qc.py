from lib.util.data import read_preprocessed_file
from lib.util.parse import get_high_dis_qc

data = read_preprocessed_file(r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2012_train_split_5.txt")
concepts_high_distinction, questions_high_distinction = get_high_dis_qc(data, "single_concept", dict())
