import os

from lib.dataset.util import parse4dataset_enhanced
from lib.util.data import read_preprocessed_file


def max_entropy_adv_aug_general_config(local_params, global_params, global_objects):
    # adv aug相关参数
    use_warm_up4cl = local_params["use_warm_up"]
    epoch_warm_up4cl = local_params["epoch_warm_up"]
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]
    weight_adv_pred_loss = local_params["weight_adv_pred_loss"]

    global_params["other"]["max_entropy_adv_aug"] = {}
    max_entropy_adv_aug_config = global_params["other"]["max_entropy_adv_aug"]
    max_entropy_adv_aug_config["use_warm_up"] = use_warm_up4cl
    max_entropy_adv_aug_config["epoch_warm_up"] = epoch_warm_up4cl
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
    enhance_method = local_params["enhance_method"]
    num_min_question4diff = local_params["num_min_question4diff"]
    hard_acc = local_params["hard_acc"]
    easy_acc = local_params["easy_acc"]
    weight_enhance_loss1 = local_params["weight_enhance_loss1"]
    weight_enhance_loss2 = local_params["weight_enhance_loss2"]
    num_few_shot = local_params["num_few_shot"]
    enhance_method2_update_few_shot = local_params["enhance_method2_update_few_shot"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt_output_enhance"
    global_params["loss_config"]["enhance loss 1"] = weight_enhance_loss1
    global_params["loss_config"]["enhance loss 2"] = weight_enhance_loss2

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))

    global_params["other"]["output_enhance"] = {
        "enhance_method": enhance_method,
        "weight_enhance_loss1": weight_enhance_loss1,
        "weight_enhance_loss2": weight_enhance_loss2,
        "num_min_question4diff": num_min_question4diff,
        "hard_acc": hard_acc,
        "easy_acc": easy_acc,
        "num_few_shot": num_few_shot,
        "enhance_method2_update_few_shot": enhance_method2_update_few_shot
    }
    concept_dict, question_dict = parse4dataset_enhanced(dataset_train,
                                                         global_objects["data"]["question2concept"],
                                                         global_objects["data"]["concept2question"],
                                                         global_params["other"]["output_enhance"])

    global_objects["kt_enhance"] = {}
    global_objects["kt_enhance"]["concept_dict"] = concept_dict
    global_objects["kt_enhance"]["question_dict"] = question_dict

    global_objects["logger"].info(
        "output enhance params\n"
        f"    enhance_method: {enhance_method}, weight of enhance loss1: {weight_enhance_loss1}, weight of enhance loss2: {weight_enhance_loss2}, "
        f"min num of question for calculate difficulty: {num_min_question4diff}\n"
        f"    accuracy of hard question: {hard_acc}, accuracy of easy question: {easy_acc},  num of few shot question: {num_few_shot}, "
        f"use enhance method2 to update few shot question: {enhance_method2_update_few_shot}"
    )
