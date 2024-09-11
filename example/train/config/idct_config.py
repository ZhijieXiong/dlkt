from ._config import *
from ._cl_config import *

from lib.template.kt_model.IDCT import MODEL_PARAMS as IDCT_MODEL_PARAMS
from lib.util.basic import *


def idct_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(IDCT_MODEL_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    max_que_disc = local_params["max_que_disc"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["IDCT"]
    encoder_config["num_question"] = num_question
    encoder_config["num_concept"] = num_concept
    encoder_config["dim_emb"] = dim_emb
    encoder_config["max_que_disc"] = max_que_disc
    encoder_config["dropout"] = dropout

    # 辅助损失
    w_monotonic = local_params.get("w_monotonic", 0)
    w_mirt = local_params.get("w_mirt", 0)
    if w_monotonic != 0:
        global_params["loss_config"]["monotonic loss"] = w_monotonic
    if w_mirt != 0:
        global_params["loss_config"]["mirt loss"] = w_mirt

    # 配置多知识点习题penalty损失权重
    Q_table = global_objects["data"]["Q_table"]
    qc_counts = Q_table.sum(axis=-1)
    num_max_qc_count = max(qc_counts)
    if num_max_qc_count > 1:
        # 损失权重1：单调下降
        loss_weight1 = torch.from_numpy(np.exp(1 - qc_counts)).to(global_params["device"])
        global_objects["data"]["loss_weight1"] = loss_weight1

        # 损失权重2：先下降后上升
        weight1 = np.exp(1 - qc_counts)
        # 考虑到一道题对应知识点越多，习题越难，那么可能就越要求每个知识点掌握程度都高，这个权重应该是先下降再上升
        th = num_max_qc_count / 2 + 0.5
        ones_array = np.ones_like(qc_counts)
        exp_neg = np.exp(ones_array - th)
        # 分母
        num1 = ones_array * (num_max_qc_count - th)
        # 分子
        num2 = (ones_array - exp_neg) * qc_counts + num_max_qc_count * exp_neg - ones_array * th
        weight2 = num2 / num1
        # 知识点数目小于3使用1，大于等于3使用2
        weight1[qc_counts > th] = 0
        weight2[qc_counts < th] = 0
        loss_weight2 = torch.from_numpy(weight1 + weight2).to(global_params["device"])
        global_objects["data"]["loss_weight2"] = loss_weight2

    global_objects["logger"].info(
        "model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_emb: {dim_emb}, dropout: {dropout}\n"
        f"loss config\n    "
        f"w_monotonic: {w_monotonic}, w_mirt: {w_mirt}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"IDCT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def idct_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    idct_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def idct_two_stage_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    idct_general_config(local_params, global_params, global_objects)

    # 配置两个优化器的参数（使用相同的参数）
    config_optimizer(local_params, global_params, global_objects, "user", same_as_kt=True)
    config_optimizer(local_params, global_params, global_objects, "question", same_as_kt=True)

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@IDCT_two_stage@@seed_{local_params['seed']}"
            f"@@{setting_name}@@{train_file_name.replace('.txt', '')}"
        )
        save_params(global_params, global_objects)

    return global_params, global_objects
