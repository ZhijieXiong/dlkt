import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from .util import *
from ..model.Model4LongTail import *


class MutualEnhance4LongTailTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(MutualEnhance4LongTailTrainer, self).__init__(params, objects)

        self.question_branch = ItemBranch(params, objects)
        head_question_dataset = TensorDataset(
            torch.LongTensor(objects["mutual_enhance4long_tail"]["head_questions"])
        )
        tail_question_dataset = TensorDataset(
            torch.LongTensor(objects["mutual_enhance4long_tail"]["tail_questions"])
        )
        num_batch = len(self.objects["data_loaders"]["train_loader"])
        self.head_question_data_loader = DataLoader(
            head_question_dataset,
            len(head_question_dataset) // (num_batch - 1) + 1,
            shuffle=True,
            drop_last=False
        )
        self.tail_question_data_loader = DataLoader(
            tail_question_dataset,
            256,
            shuffle=False,
            drop_last=False
        )

    def init_trainer(self):
        # 初始化optimizer和scheduler
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]

        kt_model = self.objects["models"]["kt_model"]
        kt_model_optimizer_config = self.params["optimizers_config"]["kt_model"]
        kt_model_scheduler_config = self.params["schedulers_config"]["kt_model"]

        kt_model_params = [{"params": kt_model.parameters()}, {"params": self.question_branch.parameters()}]
        optimizers["kt_model"] = create_optimizer(kt_model_params, kt_model_optimizer_config)
        if kt_model_scheduler_config["use_scheduler"]:
            schedulers["kt_model"] = create_scheduler(optimizers["kt_model"], kt_model_scheduler_config)
        else:
            schedulers["kt_model"] = None

    def train(self):
        train_strategy = self.params["train_strategy"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        kt_model = self.objects["models"]["kt_model"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]

        self.print_data_statics()
