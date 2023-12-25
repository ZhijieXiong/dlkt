from _config import *
from _cl_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS
from lib.template.model.DIMKT import MODEL_PARAMS as DIMKT_MODEL_PARAMS
from lib.util.basic import *
from lib.dataset.util import parse_difficulty
from lib.util.data import read_preprocessed_file


def dimkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(DIMKT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DIMKT"

    # 配置模型参数和数据集参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_min_question = local_params["num_min_question"]
    num_min_concept = local_params["num_min_concept"]
    num_question_diff = local_params["num_question_diff"]
    num_concept_diff = local_params["num_concept_diff"]
    dim_emb = local_params["dim_emb"]
    dropout = local_params["dropout"]

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    question_difficulty, concept_difficulty = \
        parse_difficulty(dataset_train, global_params["datasets_config"]["data_type"],
                         num_min_question, num_min_concept, num_question_diff, num_concept_diff)
    global_objects["dimkt"] = {}
    global_objects["dimkt"]["question_difficulty"] = question_difficulty
    global_objects["dimkt"]["concept_difficulty"] = concept_difficulty
    global_params["datasets_config"]["train"]["type"] = "kt4dimkt"

    # encoder layer
    akt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
    akt_encoder_layer_config["num_concept"] = num_concept
    akt_encoder_layer_config["num_question"] = num_question
    akt_encoder_layer_config["num_question_diff"] = num_question_diff
    akt_encoder_layer_config["num_concept_diff"] = num_concept_diff
    akt_encoder_layer_config["dim_emb"] = dim_emb
    akt_encoder_layer_config["dropout"] = dropout

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]

    print("model params\n"
          f"    num of concept: {num_concept}, num of question: {num_question}, num of min question: {num_min_question}, "
          f"num of min concept: {num_min_concept}, num of question difficulty: {num_question_diff}, "
          f"num of concept difficulty: {num_concept_diff}, dim of emb: {dim_emb}, dropout: {dropout}\n")

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@DIMKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def dimkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
