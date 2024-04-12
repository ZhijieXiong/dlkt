from copy import deepcopy
from collections import defaultdict

from ..util.parse import cal_accuracy4data, get_high_low_accuracy_seqs, cal_diff
from ..CONSTANT import FORGET_POINT, REMAIN_PERCENT


class CognitionTracingUtil:
    """
    建模学生学习和遗忘行为

    根据数据集量化学生在各个时刻对各知识点的掌握程度，遵守以下原则：

    （1）某个知识点的习题正确率越高，对该知识点的掌握程度越高

    （2）当连续两次做相同知识点的习题时间间隔较长时，最新一次的影响权重应更大（因为没有学生学习的数据，只有做题数据，所以不能判断这段时间里学生是否学习了相关知识，加大当前记录的权重）

    （3）做对一道题带来的知识点掌握程度提升应该比做错大

    （4）做对一道难题带来的知识点掌握程度提升应该比做做对一道简单题大（这里的难度比较仅限于同一知识点下）

    （5）学生对某个知识点的遗忘程度因随着做（该知识点）题频次增加而下降，甚至考虑学生已经永久掌握该知识点（LPKT的一个缺陷就是建模出来的学生遗忘太快，并且没考虑永久掌握）
    """
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.dataset_train = None
        pass

    def label_user_ability(self, dataset):
        """
        提取出user在每个时刻对指定知识点的认知标签和权重，即对于一个user，要提取出这样一条序列:\n
        [..., (t, Ct, master_score_c, weight_c_t), ...]\n

        :param dataset:
        :return:
        """
        factor_seqs = []
        que_difficulty = self.objects["cognition_tracing"]["que_difficulty"]
        que_diff_ave = sum(que_difficulty.values()) / len(que_difficulty)
        for item_data in dataset:
            factor_seq = []
            concept_record = {}
            for t in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][t]
                q_diff = que_difficulty.get(q_id, -1)
                correct = item_data["correct_seq"][t]
                time = item_data["time_seq"][t]
                c_ids = self.objects["data"]["question2concept"][q_id]
                for c_id in c_ids:
                    concept_record.setdefault(c_id, {"last_time": 0, 'N': 0, "last_master_score": 0})
                    last_time = concept_record[c_id]["last_time"]
                    last_master_score = concept_record[c_id]["last_master_score"]

                    # 类似LPKT：计算历史正确率（不包括当前），得到一个历史掌握情况m_{t-1}
                    #          计算当前做的习题能带来的提升l_t（根据做对做错以及习题难度）
                    #          计算遗忘情况f_t（根据艾宾浩斯遗忘曲线和间隔时间）
                    #          m_t = w_t * l_t + (1 - w_t) * m_{t-1} * f_t，其中w_t是根据间隔时间得到的，上一次间隔时间越大，w_t越大
                    if last_time == 0:
                        if q_diff == -1:
                            master_score = (0.5 if correct else 0.1) * que_diff_ave
                            weight = 0.05
                        else:
                            master_score = (0.5 if correct else 0.1) * q_diff
                            weight = 0.1
                    else:
                        weight = min(1, concept_record[c_id]["N"] / 10)
                        interval_time = time - last_time
                        if q_diff == -1:
                            learn = (0.5 if correct else 0.1) * que_diff_ave
                            weight *= 0.5
                        else:
                            learn = (0.5 if correct else 0.1) * q_diff

                        # 以天为单位
                        if (interval_time // (60 * 24)) > 30:
                            master_score = min(1, learn + last_master_score * 0.2)
                            weight = 0.1
                        elif (interval_time // (60 * 24)) > 7:
                            master_score = min(1, learn + last_master_score * 0.5)
                            weight = 0.2
                        else:
                            master_score = min(1, learn + last_master_score * 0.8)

                    concept_record[c_id]["last_time"] = time
                    concept_record[c_id]["N"] += 1
                    concept_record[c_id]["last_master_score"] = master_score
                    factor_item = (t, c_id, master_score, weight)
                    factor_seq.append(factor_item)
            factor_seqs.append(factor_seq)

        print("")

    def cal_question_diff(self, dataset):
        min_count2drop = self.params["other"]["cognition_tracing"]["min_fre4diff"]
        corrects = defaultdict(int)
        counts = defaultdict(int)

        for item_data in dataset:
            for i in range(item_data["seq_len"]):
                k_id = item_data["question_seq"][i]
                correct = item_data["correct_seq"][i]
                corrects[k_id] += correct
                counts[k_id] += 1

        # 丢弃练习次数少于min_count次的习题或者知识点
        all_ids = list(counts.keys())
        for k_id in all_ids:
            if counts[k_id] < min_count2drop:
                del counts[k_id]
                del corrects[k_id]

        return {k_id: corrects[k_id] / float(counts[k_id]) for k_id in corrects}

    @staticmethod
    def cal_que_discrimination(data_uniformed, params):
        # 公式：总分最高的PERCENT_THRESHOLD学生（H）和总分最低的PERCENT_THRESHOLD学生（L），计算H和L对某道题的通过率，之差为区分度
        NUM2DROP4QUESTION = params.get("min_fre4disc", 30)
        MIN_SEQ_LEN = params.get("min_seq_len4disc", 20)
        PERCENT_THRESHOLD = params.get("percent_threshold", 0.37)

        dataset_question = deepcopy(data_uniformed)
        cal_accuracy4data(dataset_question)
        H_question, L_question = get_high_low_accuracy_seqs(dataset_question, MIN_SEQ_LEN, PERCENT_THRESHOLD)
        H_question_diff = cal_diff(H_question, "question_seq", NUM2DROP4QUESTION)
        L_question_diff = cal_diff(L_question, "question_seq", NUM2DROP4QUESTION)

        intersection_H_L = set(H_question_diff.keys()).intersection(set(L_question_diff.keys()))
        res = {}
        for q_id in intersection_H_L:
            res[q_id] = H_question_diff[q_id] - L_question_diff[q_id]

        return res
