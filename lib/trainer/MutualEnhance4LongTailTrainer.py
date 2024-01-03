from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
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
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        kt_model = self.objects["models"]["kt_model"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        weight_seq_loss = self.params["loss_config"]["seq transfer loss"]
        weight_question_loss = self.params["loss_config"]["seq transfer loss"]
        use_transfer4seq = self.params["other"]["mutual_enhance4long_tail"]["use_transfer4seq"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            kt_model.eval()
            self.seq_branch.eval()
            self.question_branch.eval()
            with torch.no_grad():
                # Knowledge transfer from item branch to user branch
                for batch_tail_question in self.tail_question_data_loader:
                    kt_model.update_tail_question(batch_tail_question, self.question_branch)

            kt_model.train()
            self.seq_branch.train()
            self.question_branch.train()
            loaders = zip(train_loader, self.head_question_data_loader, self.head_seq_data_loader)
            for batch, batch_head_question, batch_head_seq in loaders:
                optimizer.zero_grad()

                num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
                num_head_q = len(batch_head_question[0])
                num_head_s = len(batch_head_seq[0])

                loss = 0.
                if use_transfer4seq:
                    seq_loss = self.seq_branch.get_transfer_loss(batch_head_seq, kt_model, epoch)
                    self.loss_record.add_loss("seq transfer loss", seq_loss.detach().cpu().item() * num_head_s, num_head_s)
                    loss = loss + weight_seq_loss * seq_loss

                question_loss = self.question_branch.get_transfer_loss(batch_head_question, kt_model, self.seq_branch, epoch)
                self.loss_record.add_loss("question transfer loss", question_loss.detach().cpu().item() * num_head_q, num_head_q)
                loss = loss + weight_question_loss * question_loss

                predict_loss = kt_model.get_predict_loss(batch)
                self.loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + predict_loss

                loss.backward()
                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(kt_model.parameters(), max_norm=grad_clip_config["grad_clipped"])

                optimizer.step()
            if schedulers_config["use_scheduler"]:
                scheduler.step()

            self.evaluate()
            if self.stop_train():
                break

    def evaluate_kt_dataset(self, model, data_loader):
        model.eval()
        self.seq_branch.eval()
        self.question_branch.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                predict_score = model.get_predict_score4long_tail(batch, self.seq_branch).detach().cpu().numpy()
                ground_truth = torch.masked_select(correct_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5
        return {"AUC": AUC, "ACC": ACC, "MAE": MAE, "RMSE": RMSE}
