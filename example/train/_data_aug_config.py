import os

from lib.dataset.util import parse4dataset_enhanced
from lib.util.data import read_preprocessed_file


def max_entropy_adv_aug_general_config(local_params, global_params, global_objects):
    # adv aug相关参数
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]
    weight_adv_pred_loss = local_params["weight_adv_pred_loss"]

    global_params["other"]["max_entropy_adv_aug"] = {}
    max_entropy_adv_aug_config = global_params["other"]["max_entropy_adv_aug"]
    max_entropy_adv_aug_config["epoch_interval_generate"] = epoch_interval_generate
    max_entropy_adv_aug_config["loop_adv"] = loop_adv
    max_entropy_adv_aug_config["epoch_generate"] = epoch_generate
    max_entropy_adv_aug_config["adv_learning_rate"] = adv_learning_rate
    max_entropy_adv_aug_config["eta"] = eta
    max_entropy_adv_aug_config["gamma"] = gamma
    global_params["loss_config"]["adv predict loss"] = weight_adv_pred_loss

    global_objects["logger"].info(
        "max entropy adversarial data augment\n"
        f"    adv lr: {adv_learning_rate}, eta: {eta}, gamma: {gamma}, weight of predict loss in augmented data: {weight_adv_pred_loss}\n"
        f"    interval epoch of generate: {epoch_interval_generate}, num of adv generation loop: {loop_adv}, num epoch of generation: {epoch_generate}"
        f""
    )


def output_enhance_general_config(local_params, global_params, global_objects):
    num_min_question4diff = local_params["num_min_question4diff"]
    hard_acc = local_params["hard_acc"]
    easy_acc = local_params["easy_acc"]
    weight_enhance_loss = local_params["weight_enhance_loss"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt_output_enhance"
    global_params["loss_config"]["enhance loss"] = weight_enhance_loss

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    concept_dict, question_dict = parse4dataset_enhanced(dataset_train,
                                                         global_params["datasets_config"]["data_type"],
                                                         num_min_question4diff,
                                                         global_objects["data"]["question2concept"],
                                                         global_objects["data"]["concept2question"],
                                                         hard_acc,
                                                         easy_acc)

    global_objects["kt_enhance"] = {}
    global_objects["kt_enhance"]["concept_dict"] = concept_dict
    global_objects["kt_enhance"]["question_dict"] = question_dict
