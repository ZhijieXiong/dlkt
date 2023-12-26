from lib.util.data import read_preprocessed_file
from lib.dataset.util import parse_long_tail


if __name__ == "__main__":
    data = read_preprocessed_file(r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2012_train_split_5.txt")
    parse_long_tail(data, "single_concept", 0.8, 5)
