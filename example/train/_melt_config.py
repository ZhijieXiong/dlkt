import os

from lib.util.data import read_preprocessed_file
from lib.dataset.util import parse_long_tail


def mutual_enhance4long_tail_general_config(local_params, global_params, global_objects):
    long_tail_threshold = local_params["long_tail_threshold"]
    min_context_seq_len = local_params["min_context_seq_len"]

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    question_context, head_questions = parse_long_tail(dataset_train,
                                                       global_params["datasets_config"]["data_type"],
                                                       long_tail_threshold,
                                                       min_context_seq_len)
    global_objects["mutual_enhance4long_tail"] = {}
    global_objects["mutual_enhance4long_tail"]["head_questions"] = head_questions
    global_objects["mutual_enhance4long_tail"]["question_context"] = question_context
