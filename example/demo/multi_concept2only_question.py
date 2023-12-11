import argparse

from lib.util.data import read_preprocessed_file, write2file, dataset_multi_concept2only_question


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_concept_data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2009_test_split_6.txt")

    args = parser.parse_args()
    params = vars(args)
    params["data_type"] = "multi_concept"

    data_multi_concept = read_preprocessed_file(params["multi_concept_data_path"])
    only_question_data_path = params["multi_concept_data_path"].replace(".txt", f"_only_question.txt")
    data_only_question = dataset_multi_concept2only_question(data_multi_concept)
    write2file(data_only_question, only_question_data_path)
