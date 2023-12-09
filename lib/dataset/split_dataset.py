import random
import os

from lib.util.data import write2file
from lib.dataset.KTDataset import KTDataset
from lib.util.data import load_json


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
    num_fold = num_train_valid - num_train

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
    num_fold = num_all // n_fold

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

    # dataset_uniformed = deepcopy(dataset_uniformed)
    datasets_train, datasets_valid, dataset_test = split1(dataset_uniformed, n_fold, test_radio, valid_radio)
    if n_fold == 1:
        pass
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]

    data_type = params["data_type"]
    # 生成pykt提出的测试多知识点数据集方法所需要的文件
    max_seq_len = params["max_seq_len"]
    # Q_table
    dataset_name = params["dataset_name"]
    Q_table = objects["file_manager"].get_q_table(dataset_name, data_type)

    # num_max_concept
    preprocessed_dir = objects["file_manager"].get_preprocessed_dir(dataset_name)
    statics_preprocessed_multi_concept = load_json(os.path.join(preprocessed_dir,
                                                                "statics_preprocessed_multi_concept.json"))
    num_max_concept = statics_preprocessed_multi_concept["num_max_concept"]
    for fold in range(n_fold):
        write2file(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write2file(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))
        if data_type == "multi_concept":
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(datasets_valid[fold], Q_table, num_max_concept, max_seq_len),
                os.path.join(setting_dir,
                             names_valid[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
    name_data_test = f"{dataset_name}_test.txt"
    write2file(dataset_test, os.path.join(setting_dir, name_data_test))
    if data_type == "multi_concept":
        write2file(
            KTDataset.dataset_multi_concept2question_pykt(dataset_test, Q_table, num_max_concept, max_seq_len),
            os.path.join(setting_dir,
                         name_data_test.replace(".txt", "_question_base4multi_concept.txt"))
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
    # 生成pykt提出的测试多知识点数据集方法所需要的文件
    max_seq_len = params["max_seq_len"]
    # Q_table
    dataset_name = params["dataset_name"]
    Q_table = objects["file_manager"].get_q_table(dataset_name, data_type)

    # num_max_concept
    preprocessed_dir = objects["file_manager"].get_preprocessed_dir(dataset_name)
    statics_preprocessed_multi_concept = load_json(os.path.join(preprocessed_dir,
                                                                "statics_preprocessed_multi_concept.json"))
    num_max_concept = statics_preprocessed_multi_concept["num_max_concept"]
    for fold in range(n_fold):
        write2file(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write2file(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))
        write2file(datasets_test[fold], os.path.join(setting_dir, names_test[fold]))
        if data_type == "multi_concept":
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(names_test[fold], Q_table, num_max_concept, max_seq_len),
                os.path.join(setting_dir,
                             names_test[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
            write2file(
                KTDataset.dataset_multi_concept2question_pykt(names_test[fold], Q_table, num_max_concept, max_seq_len),
                os.path.join(setting_dir,
                             names_test[fold].replace(".txt", "_question_base4multi_concept.txt"))
            )
