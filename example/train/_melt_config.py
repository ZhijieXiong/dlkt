import os

from lib.util.data import read_preprocessed_file
from lib.dataset.util import parse_long_tail


def mutual_enhance4long_tail_general_config(local_params, global_params, global_objects):
    head_question_threshold = local_params["head_question_threshold"]
    head_seq_len = local_params["head_seq_len"]
    min_context_seq_len = local_params["min_context_seq_len"]
    dim_question = local_params["dim_question"]
    dim_latent = local_params["dim_latent"]
    use_transfer4seq = local_params["use_transfer4seq"]
    beta4transfer_seq = local_params["beta4transfer_seq"]
    gamma4transfer_question = local_params["gamma4transfer_question"]
    max_seq_len = local_params["max_seq_len"]

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

    # 损失权重
    weight_seq_loss = local_params["weight_seq_loss"]
    weight_question_loss = local_params["weight_question_loss"]
    global_params["loss_config"]["seq transfer loss"] = weight_seq_loss
    global_params["loss_config"]["question transfer loss"] = weight_question_loss

    global_params["other"]["mutual_enhance4long_tail"] = {
        "dim_question": dim_question,
        "dim_latent": dim_latent,
        "head_question_threshold": head_question_threshold,
        "head_seq_len": head_seq_len,
        "min_context_seq_len": min_context_seq_len,
        "use_transfer4seq": use_transfer4seq,
        "beta4transfer_seq": beta4transfer_seq,
        "gamma4transfer_question": gamma4transfer_question,
        "max_seq_len": max_seq_len
    }

    global_objects["logger"].info(
        "long tail params\n"
        f"    weight of question transfer loss: {weight_question_loss}, dim of question: {dim_question}, dim of latent: {dim_latent}, max seq len: {max_seq_len}\n"
        f"    threshold of head question (percent): {head_question_threshold}, min seq len of context for question: {min_context_seq_len}, "
        f"use transfer for tail seq: {use_transfer4seq}"
        f"{f', weight of seq transfer loss: {weight_seq_loss}, seq len of head seq: {head_seq_len}, beta for transfer seq: {beta4transfer_seq}' if use_transfer4seq else ''}"
    )
