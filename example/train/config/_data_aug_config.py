import os

from lib.dataset.util import parse4dataset_enhanced
from lib.util.data import read_preprocessed_file
from lib.util.parse import question2concept_from_Q, concept2question_from_Q, cal_diff


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


def unbiased_aug_general_config(local_params, global_params, global_objects):
    num_item2unbias = local_params["num_item2unbias"]
    use_virtual_emb4question = local_params["use_virtual_emb4question"]
    use_virtual_emb4aux = local_params["use_virtual_emb4aux"]
    w_cl_loss = local_params["w_cl_loss"]
    dataset_name = local_params["dataset_name"]
    Q_table_single_concept = global_objects["file_manager"].get_q_table(dataset_name, "single_concept")
    question2concept_single_concept = question2concept_from_Q(Q_table_single_concept)
    concept2question_single_concept = concept2question_from_Q(Q_table_single_concept)

    global_params["datasets_config"]["dataset_this"] = "train"
    dataset_config = global_params["datasets_config"]["train"]
    dataset_config["kt4aug"] = {
        "type": "unbiased_aug",
        "num_aug": 1 if (w_cl_loss != 0) else 0,
        "unbiased_aug": {
            "use_virtual_emb4question": use_virtual_emb4question,
            "use_virtual_emb4aux": use_virtual_emb4aux,
            "num_question": Q_table_single_concept.shape[0],
            "num_concept_single_concept": Q_table_single_concept.shape[1],
            "num_item2unbias": num_item2unbias
        }
    }

    # 计算每道习题的正确率，找到每种知识点下最难和最简单的习题
    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    question_acc_dict = cal_diff(dataset_train, "question_seq", 10)
    most_question = {}
    for c_id in range(Q_table_single_concept.shape[1]):
        correspond_qs = concept2question_single_concept[c_id]
        if len(correspond_qs) == 0:
            # 正常数据中不可能出现
            pass
        if len(correspond_qs) == 1:
            most_question[c_id] = {
                "most_easy": correspond_qs[0],
                "most_hard": correspond_qs[0]
            }

        most_easy = correspond_qs[0]
        most_hard = correspond_qs[0]
        most_easy_acc = question_acc_dict.get(most_easy, 0)
        most_hard_acc = question_acc_dict.get(most_hard, 1)
        for q_id in correspond_qs:
            q_acc = question_acc_dict.get(q_id, -100)
            if q_acc == -100:
                continue
            if q_acc < most_hard_acc:
                most_hard = q_id
                most_hard_acc = q_acc
            if q_acc > most_easy_acc:
                most_easy = q_id
                most_easy_acc = q_acc

        most_question[c_id] = {
            "most_easy": most_easy,
            "most_hard": most_hard
        }

    global_objects["unbiased_aug"] = {
        "question2concept_single_concept": question2concept_single_concept,
        "most_question": most_question
    }

    global_objects["logger"].info(
        "data aug params\n    "
        f"num_item2unbias: {num_item2unbias}, use_virtual_emb4question: {use_virtual_emb4question}, "
        f"use_virtual_emb4aux: {use_virtual_emb4aux}"
    )
