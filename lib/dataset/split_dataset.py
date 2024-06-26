import random
import os

from lib.util.data import write2file, write_cd_task_dataset
from lib.dataset.KTDataset import KTDataset
from lib.util.data import load_json, dataset_multi_concept2only_question


def split1(dataset_uniformed, n_fold, test_radio, valid_radio, seed=0):
    """
    选一部分数据做测试集，剩余数据用n折交叉划分为训练集和验证集
    :param valid_radio:
    :param test_radio:
    :param dataset_uniformed:
    :param n_fold:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_n], [valid_fold_0, ..., valid_fold_n], test)
    """
    random.seed(seed)
    random.shuffle(dataset_uniformed)
    num_all = len(dataset_uniformed)
    num_train_valid = int(num_all * (1 - test_radio))
    num_train = int(num_train_valid * (1 - valid_radio))
    num_fold = (num_train_valid // n_fold) + 1

    if n_fold == 1:
        return dataset_uniformed[0: num_train], dataset_uniformed[num_train: num_train_valid], dataset_uniformed[num_train_valid:]

    dataset_test = dataset_uniformed[num_train_valid:]
    dataset_train_valid = dataset_uniformed[:num_train_valid]
    dataset_folds = [dataset_train_valid[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], dataset_test)
    for i in range(n_fold):
        fold_valid = i
        result[1].append(dataset_folds[fold_valid])
        folds_train = set(range(n_fold)) - {fold_valid}
        data_train = []
        for fold in folds_train:
            data_train += dataset_folds[fold]
        result[0].append(data_train)

    return result


def split2(dataset_uniformed, n_fold, valid_radio, seed=0):
    """
    先用n折交叉划分为训练集和测试集，再在训练集中划分一部分数据为验证集
    :param valid_radio:
    :param dataset_uniformed:
    :param n_fold:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_n], [valid_fold_0, ..., valid_fold_n], [test_fold_0, ..., test_fold_n])
    """
    random.seed(seed)
    random.shuffle(dataset_uniformed)
    num_all = len(dataset_uniformed)
    num_fold = (num_all // n_fold) + 1

    if n_fold <= 1:
        assert False, "num of fold must greater than 1, 5 is suggested"

    dataset_folds = [dataset_uniformed[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], [])
    for i in range(n_fold):
        fold_test = i
        result[2].append(dataset_folds[fold_test])
        folds_train_valid = set(range(n_fold)) - {fold_test}
        dataset_train_valid = []
        for fold in folds_train_valid:
            dataset_train_valid += dataset_folds[fold]
        random.shuffle(dataset_train_valid)
        num_valid = int(len(dataset_train_valid) * valid_radio)
        result[1].append(dataset_train_valid[:num_valid])
        result[0].append(dataset_train_valid[num_valid:])

    return result


def n_fold_split1(dataset_uniformed, params, objects):
    """
    第一种n折划分
    :param dataset_uniformed:
    :param params:
    :param objects:
    :return:
    """
    n_fold = params["lab_setting"]["n_fold"]
    test_radio = params["lab_setting"]["test_radio"]
    valid_radio = params["lab_setting"]["valid_radio"]
    dataset_name = params["dataset_name"]
    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])

    datasets_train, datasets_valid, dataset_test = split1(dataset_uniformed, n_fold, test_radio, valid_radio)
    if n_fold == 1:
        pass
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]

    data_type = params["data_type"]
    max_seq_len = params["max_seq_len"]
    min_seq_len = params["min_seq_len"]
    # Q_table
    dataset_name = params["dataset_name"]
    Q_table = objects["file_manager"].get_q_table(dataset_name, data_type)
    # 生成pykt提出的测试多知识点数据集方法所需要的文件
    if data_type == "multi_concept":
        # num_max_concept
        preprocessed_dir = objects["file_manager"].get_preprocessed_dir(dataset_name)
        statics_preprocessed_multi_concept = load_json(os.path.join(preprocessed_dir,
                                                                    "statics_preprocessed_multi_concept.json"))
        num_max_concept = statics_preprocessed_multi_concept["num_max_concept"]
    else:
        num_max_concept = 1

    for fold in range(n_fold):
        write2file(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write2file(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))
        if data_type == "multi_concept":
            # 对于multi concept数据，需要生成额外的数据集用于后续基于question的模型性能评估（PYKT，2022 NeurIPS提出）
            # 此外有些模型不做multi concept扩展，例如一道习题的知识点embedding是其对应的多个知识点embedding平均值
            # 这类模型需要的数据是only question
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(datasets_valid[fold], Q_table, min_seq_len, max_seq_len, num_max_concept),
                os.path.join(setting_dir,
                             names_valid[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
            write2file(
                dataset_multi_concept2only_question(datasets_train[fold], max_seq_len=max_seq_len),
                os.path.join(setting_dir,
                             names_train[fold].replace(".txt", "_only_question.txt"))
            )
            write2file(
                dataset_multi_concept2only_question(datasets_valid[fold], max_seq_len=max_seq_len),
                os.path.join(setting_dir,
                             names_valid[fold].replace(".txt", "_only_question.txt"))
            )

    name_data_test = f"{dataset_name}_test.txt"
    write2file(dataset_test, os.path.join(setting_dir, name_data_test))
    if data_type == "multi_concept":
        write2file(
            KTDataset.dataset_multi_concept2question_pykt(dataset_test, Q_table, min_seq_len, max_seq_len, num_max_concept),
            os.path.join(setting_dir,
                         name_data_test.replace(".txt", "_question_base4multi_concept.txt"))
        )
        write2file(
            dataset_multi_concept2only_question(dataset_test, max_seq_len=max_seq_len),
            os.path.join(setting_dir,
                         name_data_test.replace(".txt", "_only_question.txt"))
        )


def n_fold_split2(dataset_uniformed, params, objects):
    """
    第二种n折划分
    :param dataset_uniformed:
    :param params:
    :param objects:
    :return:
    """
    n_fold = params["lab_setting"]["n_fold"]
    valid_radio = params["lab_setting"]["valid_radio"]
    dataset_name = params["dataset_name"]
    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])

    # dataset_uniformed = deepcopy(dataset_uniformed)
    datasets_train, datasets_valid, datasets_test = split2(dataset_uniformed, n_fold, valid_radio)
    if n_fold == 1:
        pass
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]
    names_test = [f"{dataset_name}_test_fold_{fold}.txt" for fold in range(n_fold)]

    data_type = params["data_type"]
    max_seq_len = params["max_seq_len"]
    min_seq_len = params["min_seq_len"]
    # Q_table
    dataset_name = params["dataset_name"]
    Q_table = objects["file_manager"].get_q_table(dataset_name, data_type)
    # 生成pykt提出的测试多知识点数据集方法所需要的文件
    if data_type == "multi_concept":
        # num_max_concept
        preprocessed_dir = objects["file_manager"].get_preprocessed_dir(dataset_name)
        statics_preprocessed_multi_concept = load_json(os.path.join(preprocessed_dir,
                                                                    "statics_preprocessed_multi_concept.json"))
        num_max_concept = statics_preprocessed_multi_concept["num_max_concept"]
    else:
        num_max_concept = 1

    for fold in range(n_fold):
        write2file(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write2file(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))
        write2file(datasets_test[fold], os.path.join(setting_dir, names_test[fold]))
        if data_type == "multi_concept":
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(datasets_valid[fold], Q_table, min_seq_len, max_seq_len, num_max_concept),
                os.path.join(setting_dir,
                             names_valid[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(datasets_test[fold], Q_table, min_seq_len, max_seq_len, num_max_concept),
                os.path.join(setting_dir,
                             names_test[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
            write2file(
                dataset_multi_concept2only_question(datasets_train[fold], max_seq_len=max_seq_len),
                os.path.join(setting_dir,
                             names_train[fold].replace(".txt", "_only_question.txt"))
            )
            write2file(
                dataset_multi_concept2only_question(datasets_valid[fold], max_seq_len=max_seq_len),
                os.path.join(setting_dir,
                             names_valid[fold].replace(".txt", "_only_question.txt"))
            )
            write2file(
                dataset_multi_concept2only_question(datasets_test[fold], max_seq_len=max_seq_len),
                os.path.join(setting_dir,
                             names_test[fold].replace(".txt", "_only_question.txt"))
            )


def n_fold_split4CD_task1(dataset_uniformed, params, objects):
    """
    先随机划分一部分做测试集，剩下的n折划分为训练集和验证集
    :param dataset_uniformed:
    :param params:
    :param objects:
    :return:
    """
    pass


def n_fold_split4CD_task2(data4cd_task, params, objects, min_seq_len=10, seed=0):
    """
    先n折划分为训练集和测试集，再在训练集里随机划分一部分做验证集

    :param data4cd_task:
    :param params:
    :param objects:
    :param min_seq_len:
    :param seed:
    :return:
    """
    n_fold = params["lab_setting"]["n_fold"]
    if n_fold <= 1:
        assert False, "num of fold must greater than 1, 5 is suggested"
    valid_radio = params["lab_setting"]["valid_radio"]
    dataset_name = params["dataset_name"]
    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])

    random.seed(seed)
    train_datasets, valid_datasets, test_datasets = [], [], []
    for _ in range(n_fold):
        train_datasets.append([])
        valid_datasets.append([])
        test_datasets.append([])
    num_user = 0
    for user_data in data4cd_task:
        if user_data["num_interaction"] < min_seq_len:
            continue
        user_id = num_user
        num_user += 1
        all_interaction_data = user_data["all_interaction_data"]
        for interaction_data in all_interaction_data:
            interaction_data["user_id"] = user_id
        random.shuffle(all_interaction_data)

        num_fold = (user_data["num_interaction"] // n_fold) + 1
        dataset_folds = [
            all_interaction_data[num_fold * j: num_fold * (j + 1)]
            for j in range(n_fold)
        ]
        for i in range(n_fold):
            fold_test = i
            test_datasets[i] += dataset_folds[fold_test]
            folds_train_valid = set(range(n_fold)) - {fold_test}
            dataset_train_valid = []
            for fold in folds_train_valid:
                dataset_train_valid += dataset_folds[fold]
            random.shuffle(dataset_train_valid)
            num_valid = int(len(dataset_train_valid) * valid_radio)
            valid_datasets[i] += dataset_train_valid[:num_valid]
            train_datasets[i] += dataset_train_valid[num_valid:]

    with open(os.path.join(setting_dir, f"{dataset_name}_statics.txt"), "w") as f:
        f.write(f"num of user: {num_user}\n")
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]
    names_test = [f"{dataset_name}_test_fold_{fold}.txt" for fold in range(n_fold)]
    for fold in range(n_fold):
        write_cd_task_dataset(train_datasets[fold], os.path.join(setting_dir, names_train[fold]))
        write_cd_task_dataset(valid_datasets[fold], os.path.join(setting_dir, names_valid[fold]))
        write_cd_task_dataset(test_datasets[fold], os.path.join(setting_dir, names_test[fold]))

