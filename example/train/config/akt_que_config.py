from ._config import *
from lib.util.basic import *


def akt_que_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "AKT",
                "AKT": {}
            }
        }
    }

    # 配置模型参数
    dataset_name = local_params["dataset_name"]
    num_question = local_params["num_question"]
    que_emb_file_name = local_params["que_emb_file_name"]
    frozen_que_emb = local_params["frozen_que_emb"]
    dim_model = local_params["dim_model"]
    key_query_same = local_params["key_query_same"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    separate_qa = local_params["separate_qa"]
    dropout = local_params["dropout"]
    seq_representation = local_params["seq_representation"]

    # embed layer
    que_emb_path = os.path.join(
        global_objects["file_manager"].get_preprocessed_dir(dataset_name), que_emb_file_name
    )
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["question"] = {
        "num_item": num_question,
        "dim_item": dim_model,
        "learnable": not frozen_que_emb,
        "embed_path": que_emb_path
    }
    kt_embed_layer_config["question_difficulty"] = {
        "num_item": num_question,
        "dim_item": 1,
        # 对性能来说至关重要的一步
        "init_method": "zero"
    }
    kt_embed_layer_config["question_variation"] = {
        "num_item": num_question,
        "dim_item": dim_model,
    }
    kt_embed_layer_config["interaction_variation"] = {
        "num_item": 2,
        "dim_item": dim_model,
    }
    num_interaction = num_question * 2 if separate_qa else 2
    kt_embed_layer_config["interaction"] = {
        "num_item": num_interaction,
        "dim_item": dim_model,
    }

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    encoder_config["num_question"] = num_question
    encoder_config["dim_model"] = dim_model
    encoder_config["key_query_same"] = key_query_same
    encoder_config["num_head"] = num_head
    encoder_config["num_block"] = num_block
    encoder_config["dim_ff"] = dim_ff
    encoder_config["dim_final_fc"] = dim_final_fc
    encoder_config["separate_qa"] = separate_qa
    encoder_config["dropout"] = dropout
    encoder_config["seq_representation"] = seq_representation

    # 损失权重
    global_params["loss_config"]["rasch loss"] = local_params["weight_rasch_loss"]

    global_objects["logger"].info(
        f"model params\n    "
        f"num_question: {num_question}, frozen_que_emb: {frozen_que_emb}, "
        f"que_emb_path: {que_emb_path}\n    "
        f"dim_model: {dim_model}, num_block: {num_block}, num_head: {num_head}, dim_ff: {dim_ff}, "
        f"dim_final_fc: {dim_final_fc}, dropout: {dropout}, separate_qa: {separate_qa}, "
        f"key_query_same: {key_query_same}, seq_representation: {seq_representation}\n"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"AKT_QUE@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def akt_que_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    akt_que_general_config(local_params, global_params, global_objects)

    # sample_reweight
    sample_reweight = {
        "use_sample_reweight": local_params["use_sample_reweight"],
        "sample_reweight_method": local_params["sample_reweight_method"],
    }
    use_sample_reweight = local_params["use_sample_reweight"]
    sample_reweight_method = local_params["sample_reweight_method"]
    use_IPS = False
    if use_sample_reweight and local_params["sample_reweight_method"] in ["IPS-double", "IPS-seq", "IPS-question"]:
        use_IPS = True
        sample_reweight["IPS_min"] = local_params["IPS_min"]
        sample_reweight["IPS_his_seq_len"] = local_params['IPS_his_seq_len']
    global_params["sample_reweight"] = sample_reweight

    global_objects["logger"].info(
        f"sample weight\n    "
        f"use_sample_reweight: {use_sample_reweight}, sample_reweight_method: {sample_reweight_method}"
        f"{', IPS_min: ' + str(local_params['IPS_min']) + ', IPS_his_seq_len: ' + str(local_params['IPS_his_seq_len']) if use_IPS else ''}"
    )

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
