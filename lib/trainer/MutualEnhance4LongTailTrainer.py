import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from .util import *
from ..model.Model4LongTail import *


class MutualEnhance4LongTailTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        self.seq_branch = LinearSeqBranch(params, objects).to(params["device"])
        self.question_branch = LinearQuestionBranch(params, objects).to(params["device"])

        super(MutualEnhance4LongTailTrainer, self).__init__(params, objects)

        head_question_dataset = TensorDataset(
            torch.LongTensor(objects["mutual_enhance4long_tail"]["head_questions"]).to(params["device"])
        )
        tail_question_dataset = TensorDataset(
            torch.LongTensor(objects["mutual_enhance4long_tail"]["tail_questions"]).to(params["device"])
        )
        head_seq_dataset = TensorDataset(
            torch.LongTensor(objects["mutual_enhance4long_tail"]["head_seqs"]).to(params["device"])
        )
        num_batch = len(self.objects["data_loaders"]["train_loader"])
        self.head_question_data_loader = DataLoader(
            head_question_dataset,
            len(head_question_dataset) // num_batch + 1,
            shuffle=True,
            drop_last=False
        )
        self.tail_question_data_loader = DataLoader(
            tail_question_dataset,
            256,
            shuffle=False,
            drop_last=False
        )
        self.head_seq_data_loader = DataLoader(
            head_seq_dataset,
            len(head_seq_dataset) // num_batch + 1,
            shuffle=True,
            drop_last=False
        )

    def init_trainer(self):
        # 初始化optimizer和scheduler
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]

        kt_model = self.objects["models"]["kt_model"]
        kt_model_optimizer_config = self.params["optimizers_config"]["kt_model"]
        kt_model_scheduler_config = self.params["schedulers_config"]["kt_model"]

        kt_model_params = [
            {"params": kt_model.parameters()},
            {"params": self.seq_branch.parameters()},
            {"params": self.question_branch.parameters()}
        ]
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

        dataset_train = self.objects["mutual_enhance4long_tail"]["dataset_train"]
        for epoch in range(1, num_epoch + 1):
            kt_model.eval()
            with torch.no_grad():
                # Knowledge transfer from item branch to user branch
                for batch_tail_question in self.tail_question_data_loader:
                    kt_model.update_tail_question(batch_tail_question, self.question_branch)

            kt_model.train()
            loaders = zip(train_loader, self.head_question_data_loader, self.head_seq_data_loader)
            for batch, batch_head_question, batch_head_seq in loaders:
                self.seq_branch.get_transfer_loss(batch_head_seq, kt_model, epoch)
                self.question_branch.get_transfer_loss(batch_head_question, kt_model, self.seq_branch, epoch)

    def question_id2batch(self, batch_question):
        question_context = self.objects["mutual_enhance4long_tail"]["question_context"]

    def seq_id2batch(self, batch_seq):
        question_context = self.objects["mutual_enhance4long_tail"]["question_context"]
