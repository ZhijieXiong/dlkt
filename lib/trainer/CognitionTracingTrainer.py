import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .TimeRecord import TimeRecord
from ..util.basic import get_now_time


class QueDataset(Dataset):
    def __init__(self, data, device):
        self.data = torch.LongTensor(data).to(device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class CognitionTracingTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(CognitionTracingTrainer, self).__init__(params, objects)
        # self.time_record = TimeRecord()
        self.time_record = None

        self.question_concept = None
        self.prepare()

    def prepare(self):
        w_q_table = self.params["loss_config"]["q table loss"]
        if w_q_table != 0:
            question2concept = self.objects["data"]["question2concept"]
            encoder_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
            encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"][encoder_type]
            num_concept = encoder_config["num_concept"]

            self.question_concept = []
            for q_id in range(len(question2concept)):
                c_ids = question2concept[q_id]
                c_ids_ = list(set(range(num_concept)) - set(c_ids))

                related_c_ids = []
                unrelated_c_ids = []
                for c1 in c_ids:
                    for c2 in c_ids_:
                        related_c_ids.append(c1)
                        unrelated_c_ids.append(c2)

                self.question_concept.append((related_c_ids, unrelated_c_ids))

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        self.print_data_statics()
        w_que_diff_pred = self.params["loss_config"]["que diff pred loss"]
        w_que_disc_pred = self.params["loss_config"]["que disc pred loss"]
        w_q_table = self.params["loss_config"]["q table loss"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_counter_fact = self.params["loss_config"].get("counterfactual loss", 0)
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]

        num_question = len(self.objects["data"]["question2concept"])
        que_dataset = QueDataset(list(range(num_question)), self.params["device"])
        que_dataloader = DataLoader(que_dataset, batch_size=int(num_question / len(train_loader)), shuffle=True)
        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch, batch_questions in zip(train_loader, que_dataloader):
                if multi_stage:
                    optimizer.zero_grad()
                    predict_loss = model.get_predict_loss(batch, self.loss_record)
                    predict_loss.backward()
                    if grad_clip_config["use_clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                    optimizer.step()

                    if w_que_diff_pred != 0:
                        optimizer.zero_grad()
                        target_que4diff = self.objects["cognition_tracing"]["que_has_diff_ground_truth"]
                        que_diff_pred_loss = model.get_que_diff_pred_loss(target_que4diff)
                        num_que4diff = target_que4diff.shape[0]
                        self.loss_record.add_loss("que diff pred loss",
                                                  que_diff_pred_loss.detach().cpu().item() * num_que4diff, num_que4diff)
                        que_diff_pred_loss.backward()
                        if grad_clip_config["use_clip"]:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                        optimizer.step()

                    if w_que_disc_pred != 0:
                        optimizer.zero_grad()
                        target_que4disc = self.objects["cognition_tracing"]["que_has_disc_ground_truth"]
                        que_disc_pred_loss = model.get_que_disc_pred_loss(target_que4disc)
                        num_que4disc = target_que4disc.shape[0]
                        self.loss_record.add_loss("que disc pred loss",
                                                  que_disc_pred_loss.detach().cpu().item() * num_que4disc, num_que4disc)
                        que_disc_pred_loss.backward()
                        if grad_clip_config["use_clip"]:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                        optimizer.step()

                    if w_q_table != 0:
                        optimizer.zero_grad()
                        q_ids, rc_ids, urc_ids = self.iter_que4q_table_loss(batch_questions)
                        q_table_loss, num_sample = model.get_q_table_loss(batch_questions, q_ids, rc_ids, urc_ids)
                        if num_sample > 0:
                            self.loss_record.add_loss("q table loss", q_table_loss.detach().cpu().item() * num_sample,
                                                      num_sample)
                            q_table_loss.backward()
                            if grad_clip_config["use_clip"]:
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                            optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss = 0.
                    loss = loss + model.get_predict_loss(batch, self.loss_record)

                    if w_que_diff_pred != 0:
                        target_que4diff = self.objects["cognition_tracing"]["que_has_diff_ground_truth"]
                        que_diff_pred_loss = model.get_que_diff_pred_loss(target_que4diff)
                        num_que4diff = target_que4diff.shape[0]
                        self.loss_record.add_loss("que diff pred loss",
                                                  que_diff_pred_loss.detach().cpu().item() * num_que4diff, num_que4diff)
                        loss = loss + que_diff_pred_loss * w_que_diff_pred

                    if w_que_disc_pred != 0:
                        target_que4disc = self.objects["cognition_tracing"]["que_has_disc_ground_truth"]
                        que_disc_pred_loss = model.get_que_disc_pred_loss(target_que4disc)
                        num_que4disc = target_que4disc.shape[0]
                        self.loss_record.add_loss("que disc pred loss",
                                                  que_disc_pred_loss.detach().cpu().item() * num_que4disc, num_que4disc)
                        loss = loss + que_disc_pred_loss * w_que_disc_pred

                    if w_q_table != 0:
                        q_ids, rc_ids, urc_ids = self.iter_que4q_table_loss(batch_questions)
                        q_table_loss, num_sample = model.get_q_table_loss(batch_questions, q_ids, rc_ids, urc_ids)
                        if num_sample > 0:
                            self.loss_record.add_loss("q table loss", q_table_loss.detach().cpu().item() * num_sample, num_sample)
                            loss = loss + q_table_loss * w_q_table

                    loss.backward()
                    if grad_clip_config["use_clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                    optimizer.step()

            if self.time_record is not None:
                self.time_record.parse_time()

            if schedulers_config["use_scheduler"]:
                scheduler.step()

            evaluation_start = get_now_time()
            self.evaluate()
            evaluation_end = get_now_time()
            if self.time_record is not None:
                print(f"evaluation: from {evaluation_start} to {evaluation_end}")

            if self.stop_train():
                break

    def iter_que4q_table_loss(self, target_question):
        # 极其耗时
        question_ids = []
        related_concept_ids = []
        unrelated_concept_ids = []

        for i, q_id in enumerate(target_question):
            related_c_ids = self.question_concept[q_id][0]
            unrelated_c_ids = self.question_concept[q_id][1]
            question_ids += [i] * len(related_c_ids)
            related_concept_ids += related_c_ids
            unrelated_concept_ids += unrelated_c_ids

        device = self.params["device"]
        return torch.LongTensor(question_ids).to(device), \
               torch.LongTensor(related_concept_ids).to(device), \
               torch.LongTensor(unrelated_concept_ids).to(device)


