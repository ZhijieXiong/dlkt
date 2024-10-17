from ._config import *

from lib.util.basic import *


def cl4kt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "CL4KT",
                "CL4KT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    key_query_same = local_params["key_query_same"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    dropout = local_params["dropout"]

    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_model"] = dim_model
    encoder_config["key_query_same"] = key_query_same
    encoder_config["num_head"] = num_head
    encoder_config["num_block"] = num_block
    encoder_config["dim_ff"] = dim_ff
    encoder_config["dim_final_fc"] = dim_final_fc
    encoder_config["dropout"] = dropout

    # 配置对比学习
    temp = local_params["temp"]
    weight_cl_loss = local_params["weight_cl_loss"]
    use_hard_neg = local_params["use_hard_neg"]
    hard_neg_weight = local_params["hard_neg_weight"]

    global_params["loss_config"]["cl loss"] = weight_cl_loss
    global_params["other"] = {"instance_cl": {}}
    cl_config = global_params["other"]["instance_cl"]
    cl_config["temp"] = temp
    cl_config["use_hard_neg"] = use_hard_neg
    cl_config["hard_neg_weight"] = hard_neg_weight

    # 配置数据增强
    aug_order = eval(local_params["aug_order"])
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    hard_neg_prob = local_params["hard_neg_prob"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"] = {}
    datasets_train_config["kt4aug"]["aug_type"] = "random_aug"
    datasets_train_config["kt4aug"]["num_aug"] = 2
    datasets_train_config["kt4aug"]["random_aug"] = {}
    datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
    datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
    datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
    datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
    datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
    datasets_train_config["kt4aug"]["random_aug"]["use_hard_neg"] = use_hard_neg
    datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_model: {dim_model}, num_block: {num_block}, "
        f"num_head: {num_head}, dim_ff: {dim_ff}, dim_final_fc: {dim_final_fc}, key_query_same: {key_query_same}, "
        f"dropout: {dropout}"
    )

    global_objects["logger"].info(
        f"cl params\n    "
        f"temp: {temp}, weight_cl_loss: {weight_cl_loss}, use_hard_neg: {use_hard_neg}, hard_neg_weight: {hard_neg_weight}"
    )

    global_objects["logger"].info(
        f"data aug params\n    "
        f"aug_order: {local_params['aug_order']}, use_hard_neg: {use_hard_neg}, hard_neg_prob: {hard_neg_prob}, "
        f"mask_prob: {mask_prob}, crop_prob: {crop_prob}, replace_prob: {replace_prob}, permute_prob: {permute_prob}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"CL4KT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def cl4kt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    cl4kt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
