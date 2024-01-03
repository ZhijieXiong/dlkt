from lib.util.data import read_preprocessed_file
from lib.util.FileManager import FileManager
from lib.util.graph import *


if __name__ == "__main__":
    Q_table = FileManager(r"F:\code\myProjects\dlkt").get_q_table("assist2012", "single_concept")
    data = read_preprocessed_file(r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2012_train_split_5.txt")
    num_concept = 265

    concept_edge = RCD_construct_dependency_matrix(data, num_concept, Q_table)
    concept_undirected, concept_directed = RCD_process_edge(concept_edge, num_concept)
    print("")
