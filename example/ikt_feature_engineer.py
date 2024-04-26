import argparse
import itertools
import sys
import os
import inspect
import json

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from collections import defaultdict

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])


from lib.util.parse import question2concept_from_Q
from lib.util.set_up import set_seed
from lib.util.data import read_preprocessed_file
from lib.util.FileManager import FileManager


ALMOST_ONE = 0.999
ALMOST_ZERO = 0.001


def bkt_find_best_k0(X):
    res = 0.5
    kc_best = np.mean([seq[0] for seq in X])
    if kc_best > 0:
        res = kc_best

    return res


def bkt_computer_error(X, k, t, g, s):
    error = 0.0
    n = 0
    predictions = []

    for seq in X:
        current_pred = []
        pred = k
        for i, res in enumerate(seq):
            n += 1
            current_pred.append(pred)
            error += (res - pred) ** 2
            if res == 1.0:
                p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
            else:
                p = k * s / (k * s + (1 - k) * (1 - g))
            k = p + (1 - p) * t
            pred = k * (1 - s) + (1 - k) * g
        predictions.append(current_pred)

    return (error / n) ** 0.5, predictions


class BKT(BaseEstimator):
    def __init__(self, step=0.1, bounded=True, best_k0=True):
        self.k0 = ALMOST_ZERO
        self.transit = ALMOST_ZERO
        self.guess = ALMOST_ZERO
        self.slip = ALMOST_ZERO
        self.forget = ALMOST_ZERO

        self.k0_limit = ALMOST_ONE
        self.transit_limit = ALMOST_ONE
        self.guess_limit = ALMOST_ONE
        self.slip_limit = ALMOST_ONE
        self.forget_limit = ALMOST_ONE

        self.step = step
        self.best_k0 = best_k0

        if bounded:
            self.k0_limit = 0.85
            self.transit_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X):
        if self.best_k0:
            self.k0 = bkt_find_best_k0(X)
            self.k0_limit = self.k0

        k0s = np.arange(self.k0, min(self.k0_limit + self.step, ALMOST_ONE), self.step)
        transits = np.arange(self.transit, min(self.transit_limit + self.step, ALMOST_ONE), self.step)
        guesses = np.arange(self.guess, min(self.guess_limit + self.step, ALMOST_ONE), self.step)
        slips = np.arange(self.slip, min(self.slip_limit + self.step, ALMOST_ONE), self.step)
        all_parameters = [k0s, transits, guesses, slips]
        parameter_pairs = list(itertools.product(*all_parameters))

        min_error = sys.float_info.max
        for (k, t, g, s) in parameter_pairs:
            error, _ = bkt_computer_error(X, k, t, g, s)
            if error < min_error:
                self.k0 = k
                self.transit = t
                self.guess = g
                self.slip = s
                min_error = error

        return self.k0, self.transit, self.guess, self.slip


def user_id_remap(data_uniformed, id_start):
    for i, item_data in enumerate(data_uniformed):
        item_data["user_id"] = id_start + i


def bkt_predict(concept_dict, correct_dict, k, t, g, s, num_concept):
    mastery_dict = {}

    for u_id in concept_dict.keys():
        concept_seq = concept_dict[u_id]
        correct_seq = correct_dict[u_id]
        last_mastery = np.zeros(num_concept)

        if len(concept_seq) > 1:
            ini_skill = []
            all_mastery = []
            pL = np.zeros(len(concept_seq) + 1)

            for i in range(len(concept_seq)):
                if i < len(concept_seq):
                    c_id = concept_seq[i]
                    if c_id not in ini_skill:
                        ini_skill.append(c_id)
                        pL[i] = k[c_id]
                    else:
                        pL[i] = last_mastery[c_id]

                    # mastery is assessed before updating with response
                    mastery = pL[i]
                    all_mastery.append(mastery)

                    # update the mastery when response is known
                    res = correct_seq[i]
                    if res == 1.0:
                        pL[i + 1] = pL[i] * (1 - s[c_id]) / (
                                    pL[i] * (1 - s[c_id]) + (1 - pL[i]) * g[c_id])
                    else:
                        pL[i + 1] = pL[i] * s[c_id] / (pL[i] * s[c_id] + (1 - pL[i]) * (1 - g[c_id]))
                    pL[i + 1] = pL[i + 1] + (1 - pL[i + 1]) * t[c_id]
                    last_mastery[c_id] = pL[i + 1]

            mastery_dict[u_id] = all_mastery

    return mastery_dict


def get_bkt_data(data2transform, num_concept, question2concept):
    """
    bkt_dict: {concept_id: {user_id: correct_seq, ...}, ...}

    concept_dict: {user_id: concept_seq, ...}

    correct_dict: {user_id: correct_seq, ...}

    :param data2transform:
    :param num_concept:
    :param question2concept:
    :return:
    """
    bkt_dict_tmp = {}
    users_has_concept = {c_id: set() for c_id in range(num_concept)}
    concept_dict = {}
    correct_dict = {}
    question_dict = {}
    for item_data in data2transform:
        user_bkt_data = {c_id: list() for c_id in range(num_concept)}
        user_id = item_data["user_id"]
        concept_seq = []
        correct_seq = []
        question_seq = []
        for i in range(item_data["seq_len"]):
            correct = item_data["correct_seq"][i]
            q_id = item_data["question_seq"][i]
            c_ids = question2concept[q_id]
            for c_id in c_ids:
                user_bkt_data[c_id].append(correct)
                users_has_concept[c_id].add(user_id)
                concept_seq.append(c_id)
                correct_seq.append(correct)
                question_seq.append(q_id)
        concept_dict[user_id] = concept_seq
        correct_dict[user_id] = correct_seq
        question_dict[user_id] = question_seq
        bkt_dict_tmp[user_id] = user_bkt_data

    bkt_dict = {}
    for c_id in range(num_concept):
        bkt_dict[c_id] = {}
        for u_id in list(users_has_concept[c_id]):
            bkt_dict[c_id][u_id] = bkt_dict_tmp[u_id][c_id]

    return bkt_dict, concept_dict, correct_dict, question_dict


def get_q_diff_feature_dict(data_train, num_question, num_min_question, num_question_diff):
    questions_frequency = defaultdict(int)
    questions_accuracy = defaultdict(int)
    result = {}

    for item_data in data_train:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            questions_frequency[q_id] += 1
            questions_accuracy[q_id] += item_data["correct_seq"][i]

    for q_id in range(num_question):
        if questions_frequency[q_id] < num_min_question:
            result[q_id] = num_question_diff // 2
        else:
            result[q_id] = int(
                (num_question_diff - 1) * questions_accuracy[q_id] / questions_frequency[q_id])

    return result


def get_question_difficulty_feature(question_dict, q_diff_feature_dict):
    diff_feature_dict = {}
    for u_id, question_seq in question_dict.items():
        diff_feature_dict[u_id] = [q_diff_feature_dict[q_id] for q_id in question_seq]

    return diff_feature_dict


def get_ability_vector_dict(concept_dict, correct_dict, num_concept, ability_evaluate_interval=20):
    ability_vector_dict = {}
    for u_id in concept_dict.keys():
        concept_seq = concept_dict[u_id]
        correct_seq = correct_dict[u_id]

        concept_attempt = np.zeros(num_concept)
        concept_right = np.zeros(num_concept)
        for i, (c_id, correct) in enumerate(zip(concept_seq, correct_seq)):
            if (i > 0) and ((0 == i % ability_evaluate_interval) or (len(concept_seq) - 1 == i)):
                ability_vector = (concept_right + 1.4) / (concept_attempt + 2)
                seg_index = i // ability_evaluate_interval
                ability_vector_dict[str(u_id) + "-" + str(seg_index)] = ability_vector

            concept_attempt[c_id] += 1
            concept_right[c_id] += correct

    return ability_vector_dict


def get_user_ability_feature(ability_vector_dict, cluster):
    ability_feature_dict = {}
    all_keys = cluster.cluster_centers_
    for k, query in ability_vector_dict.items():
        cos_sim = query @ all_keys.T / (np.linalg.norm(query, axis=0) * np.linalg.norm(all_keys.T, axis=0) + 1e-8)
        ability_feature = np.argsort(cos_sim)[-1] + 1
        ability_feature_dict[k] = ability_feature

    return ability_feature_dict


def get_feature_all(concept_dict, correct_dict, ability_dict, q_diff_dict, mastery_dict, ability_evaluate_interval):
    feature_dict = {}
    for u_id in concept_dict.keys():
        concept_seq = concept_dict[u_id]
        correct_seq = correct_dict[u_id]
        question_diff_seq = q_diff_dict[u_id]
        mastery_seq = mastery_dict[u_id]

        seq_len = len(concept_seq)
        if seq_len <= ability_evaluate_interval:
            ability_seq = [0] * seq_len
        else:
            ability_seq = [0] * ability_evaluate_interval
            for i in range(seq_len // ability_evaluate_interval - 1):
                # 根据u_id和i找对应的画像
                ability = ability_dict[str(u_id) + "-" + str(i+1)]
                ability_seq += [ability] * ability_evaluate_interval

            if seq_len % ability_evaluate_interval != 0:
                ability = ability_dict[str(u_id) + "-" + str(seq_len // ability_evaluate_interval)]
                ability_seq += [ability] * (seq_len % ability_evaluate_interval)

        feature_dict[u_id] = {
            "concept_feature": concept_seq,
            "correct_feature": correct_seq,
            "question_diff_feature": question_diff_seq,
            "concept_mastery_feature": mastery_seq,
            "user_ability_feature": ability_seq
        }

    for u_id in feature_dict.keys():
        assert len(feature_dict[u_id]["concept_feature"]) == len(feature_dict[u_id]["user_ability_feature"]), \
            f"error: for user {u_id}, length of concept_feature != length of user_ability_feature"

    return feature_dict


def get_meta_train_data(global_params, global_objects):
    num_concept = global_params["num_concept"]
    num_question = global_params["num_question"]
    num_cluster = global_params["num_cluster"]
    ability_evaluate_interval = global_params["ability_evaluate_interval"]
    num_min_question = global_params["num_min_question"]
    num_question_diff = global_params["num_question_diff"]

    data_train = global_objects["data_train"]
    question2concept = global_objects["question2concept"]

    bkt_data_train, concept_dict_train, correct_dict_train, question_dict_train = \
        get_bkt_data(data_train, num_concept, question2concept)

    ability_vector_dict_train = \
        get_ability_vector_dict(concept_dict_train, correct_dict_train, num_concept, ability_evaluate_interval)

    # user ability meta data
    ability_vector_all = np.stack([v for v in ability_vector_dict_train.values()])
    cluster = KMeans(n_clusters=num_cluster, n_init=5, max_iter=40)
    cluster.fit(ability_vector_all)

    # question difficulty meta data
    q_diff_feature_dict = get_q_diff_feature_dict(data_train, num_question, num_min_question, num_question_diff)

    # concept mastery meta data
    DL, DT, DG, DS = {}, {}, {}, {}
    for c_id in bkt_data_train.keys():
        users_bkt_data = bkt_data_train[c_id]
        data_train4c_id = []
        for u_id in users_bkt_data.keys():
            data_train4c_id.append(users_bkt_data[u_id])

        bkt = BKT(step=0.1, bounded=False, best_k0=True)
        if len(data_train4c_id) > 2:
            DL[c_id], DT[c_id], DG[c_id], DS[c_id] = bkt.fit(data_train4c_id)
        else:
            DL[c_id], DT[c_id], DG[c_id], DS[c_id] = 0.5, 0.2, 0.1, 0.1

    return cluster, q_diff_feature_dict, {"DL": DL, "DT": DT, "DG": DG, "DS": DS}


def get_target_data_feature(global_params, global_objects, target_data="data_train"):
    num_concept = global_params["num_concept"]
    ability_evaluate_interval = global_params["ability_evaluate_interval"]

    data_target = global_objects[target_data]
    question2concept = global_objects["question2concept"]
    cluster = global_objects["train_cluster"]
    q_diff_feature_dict = global_objects["train_q_diff_dict"]
    bkt_params = global_objects["train_bkt_params"]

    _, concept_dict, correct_dict, question_dict = get_bkt_data(data_target, num_concept, question2concept)

    # 特征工程：能力画像
    ability_vector_dict = get_ability_vector_dict(concept_dict, correct_dict, num_concept, ability_evaluate_interval)
    ability_feature = get_user_ability_feature(ability_vector_dict, cluster)

    # 特征工程：习题难度
    difficulty_feature = get_question_difficulty_feature(question_dict, q_diff_feature_dict)

    # 特征工程：BKT知识追踪
    DL, DT, DG, DS = bkt_params["DL"], bkt_params["DT"], bkt_params["DG"], bkt_params["DS"]
    mastery_feature = bkt_predict(concept_dict, correct_dict, DL, DT, DG, DS, num_concept)

    feature_data = get_feature_all(
        concept_dict, correct_dict, ability_feature, difficulty_feature, mastery_feature, ability_evaluate_interval
    )

    return feature_data


def save_feature2arff(dataset_name, feature_dict, save_feature_path, min_seq_len=3):
    with open(save_feature_path, "w") as file:
        file.write(f"@relation {dataset_name}\n")
        file.write("@attribute concept_id numeric\n")
        file.write("@attribute concept_mastery numeric\n")
        file.write("@attribute ability_profile numeric\n")
        file.write("@attribute question_difficulty numeric\n")
        file.write("@attribute correctness {1,0}\n")
        file.write("@data\n")

        for feature in feature_dict.values():
            seq_len = len(feature["concept_feature"])
            if seq_len < min_seq_len:
                continue

            for i in range(1, seq_len):
                concept_feature = feature["concept_feature"][i]
                concept_mastery_feature = feature["concept_mastery_feature"][i]
                question_diff_feature = feature["question_diff_feature"][i]
                user_ability_feature = feature["user_ability_feature"][i]
                correct_feature = feature["correct_feature"][i]

                line_str = f"{int(concept_feature)},{round(concept_mastery_feature, 6)},{int(user_ability_feature)}," \
                           f"{int(question_diff_feature)},{int(correct_feature)}\n"
                file.write(line_str)


def main(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}

    setting_name = local_params["setting_name"]
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]

    num_cluster = local_params["num_cluster"]
    ability_evaluate_interval = local_params["ability_evaluate_interval"]
    num_min_question = local_params["num_min_question"]
    num_question_diff = local_params["num_question_diff"]

    # 配置参数
    global_params["setting_name"] = setting_name
    global_params["dataset_name"] = dataset_name
    global_params["data_type"] = data_type
    global_params["train_file_name"] = train_file_name
    global_params["valid_file_name"] = valid_file_name
    global_params["test_file_name"] = test_file_name

    global_params["num_concept"] = num_concept
    global_params["num_question"] = num_question
    global_params["num_cluster"] = num_cluster
    global_params["ability_evaluate_interval"] = ability_evaluate_interval
    global_params["num_min_question"] = num_min_question
    global_params["num_question_diff"] = num_question_diff

    # 读取数据
    print("Loading data ...")
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    data_train_path = os.path.join(setting_dir, train_file_name)
    data_valid_path = os.path.join(setting_dir, valid_file_name)
    data_test_path = os.path.join(setting_dir, test_file_name)

    data_train = read_preprocessed_file(data_train_path)
    if os.path.exists(data_valid_path):
        data_valid = read_preprocessed_file(data_valid_path)
    else:
        data_valid = None
    if os.path.exists(data_test_path):
        data_test = read_preprocessed_file(data_test_path)
    else:
        data_test = None

    print("User id remapping ...")
    user_id_remap(data_train, id_start=0)
    if data_valid is not None:
        user_id_remap(data_valid, id_start=len(data_train))
    if data_test is not None:
        if data_valid is not None:
            user_id_remap(data_test, id_start=len(data_train)+len(data_valid))
        else:
            user_id_remap(data_test, id_start=len(data_train))

    global_objects["data_train"] = data_train
    global_objects["data_valid"] = data_valid
    global_objects["data_test"] = data_test
    global_objects["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, data_type)
    global_objects["question2concept"] = question2concept_from_Q(global_objects["Q_table"])

    # 获取特征工程所需要的元数据，如聚类中心，bkt参数，习题难度等级
    print("Obtaining meta (user ability cluster, question difficulty level, bkt params) of train data ...")
    cluster, q_diff_feature_dict, bkt_params = get_meta_train_data(global_params, global_objects)
    global_objects["train_cluster"] = cluster
    global_objects["train_q_diff_dict"] = q_diff_feature_dict
    global_objects["train_bkt_params"] = bkt_params

    print("Feature engineering ...")
    feature_train = get_target_data_feature(global_params, global_objects, target_data="data_train")
    if data_valid is not None:
        feature_valid = get_target_data_feature(global_params, global_objects, target_data="data_valid")
    else:
        feature_valid = None
    if data_test is not None:
        feature_test = get_target_data_feature(global_params, global_objects, target_data="data_test")
    else:
        feature_test = None

    print("Saving feature data ...")
    seed = local_params["seed"]
    min_seq_len = local_params["min_seq_len"]
    feature_params_str = f"_seed-{seed}_{num_cluster}-{ability_evaluate_interval}-{num_min_question}-{num_question_diff}"

    feature_train_name = train_file_name.replace(".txt", f"_feature_{feature_params_str}.arff")
    feature_train_path = os.path.join(setting_dir, feature_train_name)
    save_feature2arff(dataset_name, feature_train, feature_train_path, min_seq_len)

    if feature_valid is not None:
        feature_valid_name = valid_file_name.replace(".txt", f"_feature_{feature_params_str}.arff")
        feature_valid_path = os.path.join(setting_dir, feature_valid_name)
        save_feature2arff(dataset_name, feature_valid, feature_valid_path, min_seq_len)

    if feature_test is not None:
        feature_test_name = test_file_name.replace(".txt", f"_feature_{feature_params_str}.arff")
        feature_test_path = os.path.join(setting_dir, feature_test_name)
        save_feature2arff(dataset_name, feature_test, feature_test_path, min_seq_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="ikt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_type", type=str, default="only_question",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="no_file")
    parser.add_argument("--test_file_name", type=str, default="assist2009_test_fold_0.txt")
    # 数据集参数
    parser.add_argument("--num_concept", type=int, default=123)
    parser.add_argument("--num_question", type=int, default=17751)
    # 特征工程参数
    parser.add_argument("--num_cluster", type=int, default=7)
    parser.add_argument("--ability_evaluate_interval", type=int, default=20)
    parser.add_argument("--num_min_question", type=int, default=4)
    parser.add_argument("--num_question_diff", type=int, default=11)
    # 其它
    parser.add_argument("--min_seq_len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    main(vars(args))
