import os

from lib.util.data import read_preprocessed_file
from lib.dataset.util import parse_long_tail


def mutual_enhance4long_tail_general_config(local_params, global_params, global_objects):
    head_question_threshold = local_params["head_question_threshold"]
    long_tail_seq_len = local_params["long_tail_seq_len"]
    head_seq_len = local_params["head_seq_len"]
    min_context_seq_len = local_params["min_context_seq_len"]
    dim_question = local_params["dim_question"]
    dim_latent = local_params["dim_latent"]

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    question_context, head_questions, head_seqs = parse_long_tail(dataset_train,
                                                                  global_params["datasets_config"]["data_type"],
                                                                  head_question_threshold,
                                                                  head_seq_len,
                                                                  min_context_seq_len)
    tail_questions = list(set(range(local_params["num_question"])) - set(head_questions))
    global_objects["mutual_enhance4long_tail"] = {}
    global_objects["mutual_enhance4long_tail"]["dataset_train"] = dataset_train
    global_objects["mutual_enhance4long_tail"]["head_questions"] = head_questions
    global_objects["mutual_enhance4long_tail"]["head_seqs"] = head_seqs
    global_objects["mutual_enhance4long_tail"]["tail_questions"] = tail_questions
    global_objects["mutual_enhance4long_tail"]["question_context"] = question_context

    global_params["other"]["mutual_enhance4long_tail"] = {
        "dim_question": dim_question,
        "dim_latent": dim_latent,
        "head_question_threshold": head_question_threshold,
        "long_tail_seq_len": long_tail_seq_len,
        "head_seq_len": head_seq_len,
        "min_context_seq_len": min_context_seq_len
    }
