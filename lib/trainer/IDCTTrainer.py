import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer

from .LossRecord import LossRecord
from .util import *


class IDCTTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def init_loss_record(self):
        used_losses = ["predict loss of user stage", "predict loss of question stage"]
        loss_config = self.params["loss_config"]
        for loss_name in loss_config:
            used_losses.append(loss_name + " of user stage")
            used_losses.append(loss_name + " of question stage")
        return LossRecord(used_losses)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]

        kt_model = self.objects["models"]["kt_model"]

        user_optimizer_config = self.params["optimizers_config"]["user"]
        question_optimizer_config = self.params["optimizers_config"]["question"]

        user_scheduler_config = self.params["schedulers_config"]["user"]
        question_scheduler_config = self.params["schedulers_config"]["question"]

        user_params = [
            {"params": kt_model.encoder_layer.parameters()},
            {"params": kt_model.embed_concept.parameters()}
        ]
        optimizers["user"] = create_optimizer(user_params, user_optimizer_config)
        if user_scheduler_config["use_scheduler"]:
            schedulers["user"] = create_scheduler(optimizers["user"], user_scheduler_config)
        else:
            schedulers["user"] = None

        question_params = [
            {"params": kt_model.embed_concept.parameters()},
            {"params": kt_model.question_diff_params},
            {"params": kt_model.question_disc_params}
        ]
        optimizers["question"] = create_optimizer(question_params, question_optimizer_config)
        if question_scheduler_config["use_scheduler"]:
            schedulers["question"] = create_scheduler(optimizers["question"], question_scheduler_config)
        else:
            schedulers["question"] = None

    def train(self):
        train_strategy = self.params["train_strategy"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]

        kt_model = self.objects["models"]["kt_model"]

        user_grad_clip_config = self.params["grad_clip_config"]["user"]
        user_schedulers_config = self.params["schedulers_config"]["user"]
        user_optimizer = self.objects["optimizers"]["user"]
        user_scheduler = self.objects["schedulers"]["user"]

        question_grad_clip_config = self.params["grad_clip_config"]["question"]
        question_schedulers_config = self.params["schedulers_config"]["question"]
        question_optimizer = self.objects["optimizers"]["question"]
        question_scheduler = self.objects["schedulers"]["question"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            kt_model.train()
            for batch_idx, batch in enumerate(train_loader):
                # --------------------------USER---------------------------
                user_optimizer.zero_grad()
                user_loss = kt_model.get_user_loss(batch, self.loss_record, user_stage=True)
                user_loss.backward()
                if user_grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(
                        [kt_model.encoder_layer.parameters(), kt_model.embed_concept.parameters()],
                        max_norm=user_grad_clip_config["grad_clipped"]
                    )
                user_optimizer.step()

                # -----------------------QUESTION---------------------------
                question_optimizer.zero_grad()
                question_loss = kt_model.get_question_loss(batch, self.loss_record, user_stage=False)
                question_loss.backward()
                if question_grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(
                        [
                            kt_model.embed_concept.parameters(),
                            kt_model.question_diff_params,
                            kt_model.question_disc_params,
                        ],
                        max_norm=question_grad_clip_config["grad_clipped"]
                    )
                question_optimizer.step()

            if user_schedulers_config["use_scheduler"]:
                user_scheduler.step()
            if question_schedulers_config["use_scheduler"]:
                question_scheduler.step()

            self.evaluate()

            if self.stop_train():
                break
