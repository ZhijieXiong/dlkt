import torch
from torch.utils.data import Dataset

from ..util.data import *
from ..util.parse import *


class KTDataset(Dataset):
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.dataset = None

        self.load_dataset()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        return result

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        dataset_config = dataset_config_this["kt"]
        unuseful_keys = dataset_config_this["unuseful_seq_keys"]
        unuseful_keys = unuseful_keys - {"seq_len"}
        base_type = dataset_config["base_type"]

        if dataset_path != "":
            dataset_original = read_preprocessed_file(dataset_path)
        else:
            dataset_original = self.objects["dataset_this"]

        id_keys, seq_keys = get_keys_from_uniform(dataset_original)
        all_keys = set(id_keys).union(seq_keys)
        id_keys = list(set(id_keys) - unuseful_keys)
        seq_keys = list(set(seq_keys) - unuseful_keys)
        unuseful_keys = all_keys - set(id_keys).union(set(seq_keys))
        for item_data in dataset_original:
            for k in unuseful_keys:
                del item_data[k]

        if data_type == "multi_concept" and base_type == "question":
            dataset_converted = self.dataset_multi_concept2question_pykt(dataset_original)
        else:
            dataset_converted = {k: [] for k in (id_keys + seq_keys)}
            if "question_seq" in seq_keys:
                dataset_converted["question_seq_mask"] = []
            if "time_seq" in seq_keys:
                dataset_converted["interval_time_seq"] = []
            for item_data in dataset_original:
                for k in id_keys:
                    dataset_converted[k].append(item_data[k])
                for k in seq_keys:
                    if data_type == "multi_concept" and k == "question_seq":
                        question_seq = item_data["question_seq"]
                        question_seq_new = []
                        current_q = question_seq[0]
                        for q in question_seq:
                            if q != -1:
                                current_q = q
                            question_seq_new.append(current_q)
                        dataset_converted["question_seq"].append(question_seq_new)
                        dataset_converted["question_seq_mask"].append(question_seq)
                    elif k == "time_seq":
                        interval_time_seq = [0]
                        for time_i in range(1, len(item_data["time_seq"])):
                            interval_time = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                            interval_time = max(0, min(interval_time, 60 * 24 * 30))
                            interval_time_seq.append(interval_time)
                        dataset_converted["interval_time_seq"].append(interval_time_seq)
                    else:
                        dataset_converted[k].append(item_data[k])
            if "time_seq" in dataset_converted.keys():
                del dataset_converted["time_seq"]
            if "question_seq_mask" in dataset_converted.keys():
                del dataset_converted["question_seq_mask"]

        for k in dataset_converted.keys():
            dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
        self.dataset = dataset_converted

    def dataset_multi_concept2question_pykt(self, dataset):
        # 假设原始习题序列为[1,2,3]，回答结果序列为[1,0,1]，习题对应知识点为{1:[5,6], 2:[7], 3:[8,9,10]}
        # 那么新生成的数据应该为
        # question_seq: [1, 0, 0, 0], concept_seq: [5, 0, 0, 0], correct_seq:  [1, 0, 0, 0], mask_seq: [1, 0, 0, 0]
        #               [1, 0, 0, 0]               [6, 0, 0, 0]                [1, 0, 0, 0]            [1, 0, 0, 0]
        #               [1, 1, 2, 0]               [5, 6, 7, 0]                [1, 1, 0, 0]            [1, 1, 1, 0]
        #               [1, 1, 2, 3]               [5, 6, 7, 8]                [1, 1, 0, 1]            [1, 1, 1, 1]
        #               [1, 1, 2, 3]               [5, 6, 7, 9]                [1, 1, 0, 1]            [1, 1, 1, 1]
        #               [1, 1, 2, 3]               [5, 6, 7, 10]               [1, 1, 0, 1]            [1, 1, 1, 1]
        # 得到的predict score应该是，其中y(i,j)表示第i道习题对应的第j个知识点的预测得分
        # [y(1,1)]
        # [y(1,2)]
        # [y(1,1), y(2,1)]
        # [y(1,2), y(2,1), y(3,1)]
        # [y(1,1), y(2,1), y(3,2)]
        # [y(1,2), y(2,1), y(3,3)]
        # 那么就可以做融合了，即late fusion
        # 需要用一个来东西来指示每个序列取第几列，如下
        # [1, 1, 2, 3, 3, 3] 表示y(1,1)和y(1,2)是习题1的，y(2,1)是习题2的，y(3,1)、y(3,2)和y(3,3)是习题3的
        dataset = dataset_delete_pad(dataset)
        dataset = dataset_agg_concept(dataset)

        interaction_index_seq = []
        interaction_index = -1
        info_names = []
        for key in dataset[0].keys():
            if type(dataset[0][key]) == list:
                info_names.append(key)
        info_names = ["question_seq"] + list(set(info_names) - {"question_seq"})
        dataset_converted = {info_name: [] for info_name in info_names}
        for i, user_data in enumerate(dataset):
            # 每个用户数据进来都要初始化
            # 初始化为[[]]是因为在第一个用户的第一道习题时，要用到
            # 即c_seq_this = dataset_converted["concept_seq"][-1] + [c_id]，当刚开始循环时
            for k, info_name in enumerate(info_names):
                dataset_converted[info_name].append([])
            interaction_index_seq.append(-1)
            seq_all = [user_data[info_name] for info_name in info_names]
            last_num_c_id = 0
            last_c_ids = []
            for j, ele_all in enumerate(zip(*seq_all)):
                # j表示用户的第j道习题
                interaction_index += 1
                q_id = ele_all[0]
                c_ids = get_concept_from_question(self.objects["Q_table"], q_id)
                num_c_id = len(c_ids)
                is_new_seq = not len(dataset_converted["correct_seq"][-1])
                for position_c_in_q, c_id in enumerate(c_ids):
                    interaction_index_seq.append(interaction_index)
                    for k, info_name in enumerate(info_names):
                        if info_name != "concept_seq":
                            if position_c_in_q == 0 and is_new_seq:
                                # 新序列开始
                                not_c_seq_this = dataset_converted[info_name][-1] + [ele_all[k]]
                            elif position_c_in_q == 0 and not is_new_seq:
                                # 一道新习题记录
                                last_record = dataset_converted[info_name][-1][-1]
                                not_c_seq_this = \
                                    dataset_converted[info_name][-1][:-1] + [last_record] * last_num_c_id + [ele_all[k]]
                            else:
                                # 习题的非第一个知识点
                                not_c_seq_this = dataset_converted[info_name][-1][:-1] + [ele_all[k]]
                            dataset_converted[info_name].append(not_c_seq_this)
                        else:
                            if position_c_in_q == 0 and is_new_seq:
                                c_seq_this = dataset_converted["concept_seq"][-1] + [c_id]
                            elif position_c_in_q == 0 and not is_new_seq:
                                c_seq_this = dataset_converted["concept_seq"][-1][:-1] + last_c_ids + [c_id]
                            else:
                                c_seq_this = dataset_converted["concept_seq"][-1][:-1] + [c_id]
                            dataset_converted["concept_seq"].append(c_seq_this)
                last_c_ids = c_ids
                last_num_c_id = num_c_id
                # current_seq_len表示当前序列长度，当这个长度大于200时，要截断
                # 也就是一个用户可能第60道习题时，知识点序列长度就已经超过200了，需要截断重新构造序列
                current_seq_len = len(dataset_converted["correct_seq"][-1])
                # 防止序列超过200长度
                # todo: num_max_concept看一下放到哪里好
                if current_seq_len >= (200 - self.params["num_max_concept"]):
                    for info_name in info_names:
                        dataset_converted[info_name].append([])
                    interaction_index_seq.append(-1)
                    last_num_c_id = 1
                    last_c_ids = []
        for info_name in info_names:
            dataset_converted[info_name] = list(filter(lambda seq: len(seq) != 0, dataset_converted[info_name]))
        interaction_index_seq = list(filter(lambda idx: idx != -1, interaction_index_seq))
        for i, correct_seq in enumerate(dataset_converted["correct_seq"]):
            if len(correct_seq) < 3:
                interaction_index_seq[i] = -1
                for info_name in info_names:
                    dataset_converted[info_name][i] = []
        for info_name in info_names:
            dataset_converted[info_name] = list(filter(lambda seq: len(seq) != 0, dataset_converted[info_name]))
        interaction_index_seq = list(filter(lambda idx: idx != -1, interaction_index_seq))
        # max_seq_len_in_result = max(map(lambda seq: len(seq), dataset_converted["correct_seq"]))
        max_seq_len_in_result = 200
        for info_name in info_names:
            for i, seq_data in enumerate(dataset_converted[info_name]):
                seq_len = len(seq_data)
                dataset_converted[info_name][i] += [0] * (max_seq_len_in_result - seq_len)
        dataset_converted_result = {info_name: [] for info_name in info_names}
        for info_name in info_names:
            dataset_converted_result[info_name] = (
                torch.tensor(dataset_converted[info_name], device=self.params["device"], dtype=torch.int64))
        dataset_converted_result["interaction_index_seq"] = (
            torch.tensor(interaction_index_seq, device=self.params["device"], dtype=torch.int64))
        return dataset_converted_result

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correct_seq = self.dataset["correct_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correct_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction
