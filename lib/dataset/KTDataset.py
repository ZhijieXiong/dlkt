from torch.utils.data import Dataset
import random

from ..util.data import *
from ..util.parse import *
from .sample_reweight import *


class KTDataset(Dataset):
    def __init__(self, params, objects):
        super(KTDataset, self).__init__()
        self.params = params
        self.objects = objects
        self.dataset = None

        self.load_dataset()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        use_mix_up = self.params.get("use_mix_up", False)
        if use_mix_up:
            question_seq4mix_up = []
            mask_seq4mix_up = []

            question_seq = self.dataset["question_seq"][index].cpu().detach().tolist()
            correct_seq = self.dataset["correct_seq"][index].cpu().detach().tolist()
            mask_seq = self.dataset["mask_seq"][index].cpu().detach().tolist()
            max_seq_len = len(question_seq)

            # 从历史序列中随机找一个标签（如果有的话）的习题，作为增强样本
            his_q_right = []
            his_q_wrong = []
            for i, (q_id, c, m) in enumerate(zip(question_seq, correct_seq, mask_seq)):
                if m == 0:
                    break
                question4mix_up = 0
                mask4mix_up = 0

                if (c == 1) and (len(his_q_right) > 0):
                    question4mix_up = random.choice(his_q_right)
                    mask4mix_up = 1

                if (c == 0) and (len(his_q_wrong) > 0):
                    question4mix_up = random.choice(his_q_wrong)
                    mask4mix_up = 1

                if c == 0:
                    his_q_wrong.append(q_id)
                else:
                    his_q_right.append(q_id)

                question_seq4mix_up.append(question4mix_up)
                mask_seq4mix_up.append(mask4mix_up)

            question_seq4mix_up += [0] * (max_seq_len - len(question_seq4mix_up))
            mask_seq4mix_up += [0] * (max_seq_len - len(mask_seq4mix_up))

            result["question_seq4mix_up"] = torch.LongTensor(question_seq4mix_up).to(self.params["device"])
            result["mask_seq4mix_up"] = torch.LongTensor(mask_seq4mix_up).to(self.params["device"])

        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        return result

    def load_dataset(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        data_type = self.params["datasets_config"]["data_type"]
        dataset_type = dataset_config_this.get("type", "kt")
        if dataset_type == "agg_aux_info":
            agg_num = dataset_config_this["agg_aux_info"].get("agg_num", False)
        else:
            agg_num = False
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        unuseful_keys = dataset_config_this.get("unuseful_seq_keys", {"user_id"})
        unuseful_keys = unuseful_keys - {"seq_len"}

        if dataset_path != "":
            dataset_original = read_preprocessed_file(dataset_path)
        else:
            dataset_original = self.objects["dataset_this"]

        if dataset_type == "kt4dimkt":
            question_difficulty = self.objects["dimkt"]["question_difficulty"]
            concept_difficulty = self.objects["dimkt"]["concept_difficulty"]
            qc_difficulty = (question_difficulty, concept_difficulty)
            KTDataset.parse_difficulty(dataset_original, data_type, qc_difficulty)

        id_keys, seq_keys = get_keys_from_uniform(dataset_original)
        all_keys = set(id_keys).union(seq_keys)
        id_keys = list(set(id_keys) - unuseful_keys)
        seq_keys = list(set(seq_keys) - unuseful_keys)
        unuseful_keys = all_keys - set(id_keys).union(set(seq_keys))
        for item_data in dataset_original:
            for k in unuseful_keys:
                del item_data[k]

        dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        if "question_seq" in seq_keys:
            dataset_converted["question_seq_mask"] = []
        if "time_seq" in seq_keys:
            dataset_converted["interval_time_seq"] = []
        # 根据num_hint和num_attempt取值分为12类，其中num_hint可选值为unknown、0、1、multi，num_attempt可选值为unknown、1、multi
        dataset_converted["hint_attempt_seq"] = []
        dataset_converted["seq_id"] = []
        if self.params.get("sample_reweight", False) and self.params["sample_reweight"].get("use_sample_reweight", False):
            dataset_converted["weight_seq"] = []

        max_seq_len = len(dataset_original[0]["mask_seq"])
        for seq_i, item_data in enumerate(dataset_original):
            seq_len = item_data["seq_len"]

            hint_attempt_seq = []
            for i in range(seq_len):
                if "num_attempt_seq" in seq_keys and "num_hint_seq" in seq_keys:
                    num_attempt = item_data["num_attempt_seq"][i]
                    num_hint = item_data["num_hint_seq"][i]
                    if num_attempt == 1:
                        hint_count = 6 + (num_hint if num_hint <= 1 else 2)
                    else:
                        hint_count = 9 + (num_hint if num_hint <= 1 else 2)
                elif "num_attempt_seq" in seq_keys:
                    num_attempt = item_data["num_attempt_seq"][i]
                    hint_count = 1 if num_attempt == 1 else 2
                elif "num_hint_seq" in seq_keys:
                    num_hint = item_data["num_hint_seq"][i]
                    hint_count = 3 + (num_hint if num_hint <= 1 else 2)
                else:
                    hint_count = 0
                hint_attempt_seq.append(hint_count)
            hint_attempt_seq += [0] * (max_seq_len - seq_len)
            dataset_converted["hint_attempt_seq"].append(hint_attempt_seq)

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
                    for time_i in range(1, seq_len):
                        interval_time_real = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time_idx = max(0, min(interval_time_real, 60 * 24 * 30))
                        interval_time_seq.append(interval_time_idx)
                    interval_time_seq += [0] * (max_seq_len - seq_len)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                elif k == "use_time_seq":
                    dataset_converted["use_time_seq"].append(item_data["use_time_seq"])
                elif k in ["num_hint_seq", "num_attempt_seq"] and agg_num:
                    num_seq = list(map(lambda x: x if (x <= 10) else (
                            (5 + x // 5) if (x <= 50) else (50 + x // 10)
                    ), item_data[k]))
                    dataset_converted[k].append(num_seq)
                else:
                    dataset_converted[k].append(item_data[k])

            # 生成weight seq
            if self.params.get("sample_reweight", False) and self.params["sample_reweight"].get("use_sample_reweight", False):
                if self.params["sample_reweight"]["sample_reweight_method"] == "IPS-seq":
                    IPS_min = self.params["sample_reweight"]["IPS_min"]
                    IPS_his_seq_len = self.params["sample_reweight"]["IPS_his_seq_len"]
                    w_seq = IPS_seq_weight(item_data, IPS_min, IPS_his_seq_len)
                elif self.params["sample_reweight"]["sample_reweight_method"] == "IPS-question":
                    IPS_min = self.params["sample_reweight"]["IPS_min"]
                    w_seq = IPS_question_weight(item_data, self.objects["data"]["train_data_statics_common"], IPS_min)
                elif self.params["sample_reweight"]["sample_reweight_method"] == "IPS-double":
                    IPS_min = self.params["sample_reweight"]["IPS_min"]
                    IPS_his_seq_len = self.params["sample_reweight"]["IPS_his_seq_len"]
                    w_seq = IPS_double_weight(item_data, self.objects["data"]["train_data_statics_common"],
                                              IPS_min, IPS_his_seq_len)
                else:
                    raise NotImplementedError()
                dataset_converted["weight_seq"].append(w_seq)
            dataset_converted["seq_id"].append(seq_i)

        if "time_seq" in dataset_converted.keys():
            del dataset_converted["time_seq"]
        if "question_seq_mask" in dataset_converted.keys():
            del dataset_converted["question_seq_mask"]

        question2concept_combination = self.objects["data"].get("question2concept_combination", None)
        if question2concept_combination is not None:
            dataset_converted["concept_combination_seq"] = []
            for question_seq in dataset_converted["question_seq"]:
                concept_combination_seq = list(map(lambda q_id: question2concept_combination[q_id][0], question_seq))
                dataset_converted["concept_combination_seq"].append(concept_combination_seq)

        for k in dataset_converted.keys():
            if k not in ["weight_seq", "hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "correct_float"]:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.params["device"])
            else:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).float().to(self.params["device"])
        self.dataset = dataset_converted

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correct_seq = self.dataset["correct_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correct_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction

    @staticmethod
    def parse_difficulty(data_uniformed, data_type, qc_difficulty):
        if "question_diff_seq" in data_uniformed[0].keys():
            # 如果是用的dimkt_setting处理的数据，习题和知识点难度已经预处理好了
            return
        # 目前只考虑single concept数据集
        question_difficulty, concept_difficulty = qc_difficulty
        for item_data in data_uniformed:
            item_data["question_diff_seq"] = []
            for q_id in item_data["question_seq"]:
                item_data["question_diff_seq"].append(question_difficulty[q_id])

            if data_type != "only_question":
                item_data["concept_diff_seq"] = []
                for c_id in item_data["concept_seq"]:
                    item_data["concept_diff_seq"].append(concept_difficulty[c_id])

    @staticmethod
    def dataset_multi_concept2question_pykt(dataset, Q_table, min_seq_len, max_seq_len, num_max_concept):
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
        dataset = dataset_agg_concept(dataset)

        interaction_index_seq = []
        interaction_index = -1
        seq_keys = []
        for key in dataset[0].keys():
            if type(dataset[0][key]) == list:
                seq_keys.append(key)
        seq_keys = ["question_seq"] + list(set(seq_keys) - {"question_seq"})
        dataset_converted = {info_name: [] for info_name in seq_keys}
        for i, user_data in enumerate(dataset):
            # 每个用户数据进来都要初始化
            # 初始化为[[]]是因为在第一个用户的第一道习题时，要用到
            # 即c_seq_this = dataset_converted["concept_seq"][-1] + [c_id]，当刚开始循环时
            for k, info_name in enumerate(seq_keys):
                dataset_converted[info_name].append([])
            interaction_index_seq.append(-1)
            seq_all = [user_data[info_name] for info_name in seq_keys]
            last_num_c_id = 0
            last_c_ids = []
            for j, ele_all in enumerate(zip(*seq_all)):
                # j表示用户的第j道习题
                interaction_index += 1
                q_id = ele_all[0]
                c_ids = get_concept_from_question(q_id, Q_table)
                num_c_id = len(c_ids)
                is_new_seq = not len(dataset_converted["correct_seq"][-1])
                for position_c_in_q, c_id in enumerate(c_ids):
                    interaction_index_seq.append(interaction_index)
                    for k, info_name in enumerate(seq_keys):
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
                if current_seq_len >= (max_seq_len - num_max_concept):
                    for info_name in seq_keys:
                        dataset_converted[info_name].append([])
                    interaction_index_seq.append(-1)
                    last_num_c_id = 1
                    last_c_ids = []
                    
        for info_name in seq_keys:
            dataset_converted[info_name] = list(filter(lambda seq: len(seq) != 0, dataset_converted[info_name]))
            
        interaction_index_seq = list(filter(lambda idx: idx != -1, interaction_index_seq))
        for i, correct_seq in enumerate(dataset_converted["correct_seq"]):
            if len(correct_seq) < min_seq_len:
                interaction_index_seq[i] = -1
                for info_name in seq_keys:
                    dataset_converted[info_name][i] = []
        for info_name in seq_keys:
            dataset_converted[info_name] = list(filter(lambda seq: len(seq) != 0, dataset_converted[info_name]))
        interaction_index_seq = list(filter(lambda idx: idx != -1, interaction_index_seq))
        
        data_uniformed_question_base4multi_concept = []
        seq_keys = dataset_converted.keys()
        for i in range(len(interaction_index_seq)):
            item_data = {"interaction_index_seq": interaction_index_seq[i]}
            seq_len = len(dataset_converted["correct_seq"][i])
            for seq_key in seq_keys:
                item_data[seq_key] = dataset_converted[seq_key][i] + [0] * (max_seq_len - seq_len)
            item_data["seq_len"] = seq_len
            data_uniformed_question_base4multi_concept.append(item_data)

        return data_uniformed_question_base4multi_concept
