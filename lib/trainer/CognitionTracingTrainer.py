import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .TimeRecord import TimeRecord


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
        self.time_record = TimeRecord()

        self.question_concept = None
        w_q_table = self.params["loss_config"].get("q table loss", 0)
        if w_q_table != 0:
            self.prepare_question_data()
        use_pretrain = params["other"]["cognition_tracing"]["use_pretrain"]
        if use_pretrain:
            self.objects["logger"].info("\npretraining question embedding ...")
            self.pretrain_question_embed()
            self.objects["logger"].info("pretraining initial user ability ...")
            self.pretrain_encoder()

    def prepare_question_data(self):
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

    def pretrain_encoder(self):
        optimizer = self.objects["optimizers"]["kt_model"]
        model = self.objects["models"]["kt_model"]
        optimizer.zero_grad()

        user_ability_target = self.objects["cognition_tracing"]["user_ability_init"]
        for epoch in range(1, 30):
            model.train()
            user_ability_init = model.get_user_ability_init()
            loss = torch.nn.functional.mse_loss(user_ability_init, user_ability_target)
            loss.backward()
            optimizer.step()

    def pretrain_question_embed(self):
        optimizer = self.objects["optimizers"]["kt_model"]
        model = self.objects["models"]["kt_model"]
        optimizer.zero_grad()

        num_question = len(self.objects["data"]["question2concept"])
        que_dataset = QueDataset(list(range(num_question)), self.params["device"])
        que_dataloader = DataLoader(que_dataset, batch_size=128, shuffle=True)
        Q_table = self.objects["data"]["Q_table_tensor"]
        for epoch in range(1, 30):
            model.train()
            for batch_question in que_dataloader:
                optimizer.zero_grad()
                mask1 = Q_table[batch_question].bool()
                mask2 = (1-Q_table[batch_question]).bool()

                que_diff_predict = model.get_question_diff(batch_question)
                que_diff_predict1 = torch.masked_select(que_diff_predict, mask1)
                que_diff_predict2 = torch.masked_select(que_diff_predict, mask2)

                que_diff_label = Q_table[batch_question] * 0.5 + 0.05
                que_diff_label[que_diff_label > 0.5] = 0.5
                que_diff_label1 = torch.ones_like(que_diff_predict1).float().to(self.params["device"]) * 0.5
                que_diff_label2 = torch.zeros_like(que_diff_predict2).float().to(self.params["device"]) + 0.05

                loss1 = torch.nn.functional.mse_loss(que_diff_predict1, que_diff_label1)
                loss2 = torch.nn.functional.mse_loss(que_diff_predict2, que_diff_label2)
                loss = (loss1 + loss2) / 2

                loss.backward()
                optimizer.step()

    def multi_stage_train(self, batch, batch_question=None):
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        optimizer = self.objects["optimizers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        w_que_diff_pred = self.params["loss_config"].get("que diff pred loss", 0)
        w_que_disc_pred = self.params["loss_config"].get("que disc pred loss", 0)
        w_q_table = self.params["loss_config"].get("q table loss", 0)
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_counter_fact = self.params["loss_config"].get("counterfactual loss", 0)
        w_unbias_loss = self.params["loss_config"].get("w_unbias_loss", 0)

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
            que_diff_pred_loss = que_diff_pred_loss * w_que_diff_pred
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
            que_disc_pred_loss = que_disc_pred_loss * w_que_disc_pred
            que_disc_pred_loss.backward()
            if grad_clip_config["use_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
            optimizer.step()

        if w_q_table != 0:
            optimizer.zero_grad()
            q_ids, rc_ids, urc_ids = self.iter_que4q_table_loss(batch_question)
            q_table_loss, num_sample = model.get_q_table_loss(batch_question, q_ids, rc_ids, urc_ids)
            if num_sample > 0:
                self.loss_record.add_loss("q table loss", q_table_loss.detach().cpu().item() * num_sample,
                                          num_sample)
                q_table_loss = q_table_loss * w_q_table
                q_table_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

        if w_penalty_neg != 0:
            optimizer.zero_grad()
            penalty_neg_loss, num_sample = model.get_penalty_neg_loss(batch)
            if num_sample > 0:
                self.loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                          num_sample)
                penalty_neg_loss = penalty_neg_loss * w_penalty_neg
                penalty_neg_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

        if w_learning != 0:
            optimizer.zero_grad()
            learn_loss, num_sample = model.get_learn_loss(batch)
            if num_sample > 0:
                self.loss_record.add_loss("learning loss", learn_loss.detach().cpu().item() * num_sample,
                                          num_sample)
                learn_loss = learn_loss * w_learning
                learn_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

        if w_counter_fact != 0:
            optimizer.zero_grad()
            counter_fact_loss, num_sample = model.get_counter_fact_loss(batch)
            if num_sample > 0:
                self.loss_record.add_loss("counterfactual loss", counter_fact_loss.detach().cpu().item() * num_sample,
                                          num_sample)
                counter_fact_loss = counter_fact_loss * w_counter_fact
                counter_fact_loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()

        if w_unbias_loss != 0:
            batch_size = batch["mask_seq"].shape[0]
            optimizer.zero_grad()
            unbias_loss = model.get_unbias_loss(batch)
            self.loss_record.add_loss("counterfactual loss", unbias_loss.detach().cpu().item() * batch_size, batch_size)
            unbias_loss = unbias_loss * w_unbias_loss
            unbias_loss.backward()
            if grad_clip_config["use_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
            optimizer.step()

    def single_stage_train(self, batch, batch_question):
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        optimizer = self.objects["optimizers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        w_que_diff_pred = self.params["loss_config"].get("que diff pred loss", 0)
        w_que_disc_pred = self.params["loss_config"].get("que disc pred loss", 0)
        w_q_table = self.params["loss_config"].get("q table loss", 0)

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
            q_ids, rc_ids, urc_ids = self.iter_que4q_table_loss(batch_question)
            q_table_loss, num_sample = model.get_q_table_loss(batch_question, q_ids, rc_ids, urc_ids)
            if num_sample > 0:
                self.loss_record.add_loss("q table loss", q_table_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + q_table_loss * w_q_table

        loss.backward()
        if grad_clip_config["use_clip"]:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
        optimizer.step()

    def train(self):
        train_strategy = self.params["train_strategy"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        self.print_data_statics()
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]

        num_question = len(self.objects["data"]["question2concept"])
        que_dataset = QueDataset(list(range(num_question)), self.params["device"])
        que_dataloader = DataLoader(que_dataset, batch_size=int(num_question / len(train_loader)), shuffle=True)
        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch, batch_question in zip(train_loader, que_dataloader):
                if multi_stage:
                    self.multi_stage_train(batch, batch_question)
                else:
                    self.single_stage_train(batch, batch_question)

            if schedulers_config["use_scheduler"]:
                scheduler.step()

            self.evaluate()

            if self.stop_train():
                break

    def iter_que4q_table_loss(self, target_question):
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
