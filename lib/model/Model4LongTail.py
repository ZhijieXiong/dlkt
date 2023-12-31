import torch
import numpy as np
import torch.nn as nn

from ..util.data import context2batch, batch_item_data2batch


class LinearSeqBranch(nn.Module):
    def __init__(self, params, objects):
        super(LinearSeqBranch, self).__init__()
        self.params = params
        self.objects = objects

        dim_latent = params["other"]["mutual_enhance4long_tail"]["dim_latent"]

        self.W = nn.Linear(dim_latent, dim_latent)
        nn.init.xavier_normal_(self.W.weight.data)

    def get_latent_transferred(self, latent):
        return self.W(latent)

    def get_transfer_loss(self, batch_seq, kt_model, epoch):
        num_epoch = self.params["train_strategy"]["num_epoch"]
        dataset_train = self.objects["mutual_enhance4long_tail"]["dataset_train"]
        device = self.params["device"]
        max_seq_len = self.params["other"]["mutual_enhance4long_tail"]["max_seq_len"]
        head_seq_len = self.params["other"]["mutual_enhance4long_tail"]["head_seq_len"]

        full_seq = []
        part_seq = []
        weight_list = []
        Rs = np.random.randint(5, head_seq_len-1, len(batch_seq[0]))
        for R, seq_id in zip(Rs, batch_seq[0].cpu().tolist()):
            # calculate the loss coefficient
            item_data = dataset_train[seq_id]
            seq_len = item_data["seq_len"]
            weight = (np.pi / 2) * (epoch / num_epoch) + \
                     (np.pi / (2 * (max_seq_len - head_seq_len))) * (seq_len - head_seq_len)
            weight_list.append(np.abs(np.sin(weight)))

            full_seq.append(dataset_train[seq_id])
            item_data_part = {}
            for k in item_data.keys():
                if type(item_data[k]) is list:
                    item_data_part[k] = item_data[k][seq_len-R:]
                else:
                    item_data_part[k] = item_data[k]
            item_data_part["seq_len"] = R
            part_seq.append(item_data_part)

        full_batch = batch_item_data2batch(full_seq, device)
        part_batch = batch_item_data2batch(part_seq, device)

        full_latent = kt_model.get_latent_last(full_batch)
        part_latent = kt_model.get_latent_last(part_batch)

        weight_list = torch.FloatTensor(weight_list).view(-1, 1).to(self.params["device"])
        transfer_loss = (weight_list * (self.W(part_latent) - full_latent) ** 2).mean()

        return transfer_loss


class LinearQuestionBranch(nn.Module):
    def __init__(self, params, objects):
        super(LinearQuestionBranch, self).__init__()
        self.params = params
        self.objects = objects

        dim_question = params["other"]["mutual_enhance4long_tail"]["dim_question"]
        dim_latent = params["other"]["mutual_enhance4long_tail"]["dim_latent"]

        self.W4right = nn.Linear(dim_latent, dim_question)
        self.W4wrong = nn.Linear(dim_latent, dim_question)
        nn.init.xavier_normal_(self.W4right.weight.data)
        nn.init.xavier_normal_(self.W4wrong.weight.data)

        self.num_threshold = min(list(map(
            lambda q_id: len(objects["mutual_enhance4long_tail"]["question_context"][q_id]),
            objects["mutual_enhance4long_tail"]["head_questions"]
        )))

    def forward(self, batch, kt_model, seq_branch, wright_branch=True):
        use_transfer4seq = self.params["other"]["mutual_enhance4long_tail"]["use_transfer4seq"]
        beta4transfer_seq = self.params["other"]["mutual_enhance4long_tail"]["beta4transfer_seq"]
        latent = kt_model.get_latent_last(batch)
        if use_transfer4seq:
            latent = latent + beta4transfer_seq * seq_branch.get_latent_transferred(latent)
        if wright_branch:
            return self.W4right(latent)
        else:
            return self.W4wrong(latent)

    def get_question_emb_transferred(self, latent, wright_branch=True):
        if wright_branch:
            return self.W4right(latent)
        else:
            return self.W4wrong(latent)

    def get_transfer_loss(self, batch_question, kt_model, seq_branch, epoch):
        num_epoch = self.params["train_strategy"]["num_epoch"]
        question_context = self.objects["mutual_enhance4long_tail"]["question_context"]

        right_context_batch = []
        wrong_context_batch = []
        right_idx_list = [0]
        wrong_idx_list = [0]
        weight_list = []
        idx4right = 0
        idx4wrong = 0
        for q_id in batch_question[0].cpu().tolist():
            # calculate the loss coefficient
            n_context = len(question_context[q_id])
            weight = (np.pi / 2) * (epoch / num_epoch) + (np.pi / 100) * (n_context - self.num_threshold)
            weight_list.append(np.abs(np.sin(weight)))

            right_context_list = []
            wrong_context_list = []
            for q_context in question_context[q_id]:
                if q_context["correct"] == 1:
                    right_context_list.append(q_context)
                else:
                    wrong_context_list.append(q_context)

            # 随机挑选一部分context用于训练，还要考虑right或者wrong为零的情况
            threshold4right = min(len(right_context_list), self.num_threshold)
            if threshold4right > 1:
                num_context4right = np.random.randint(1, threshold4right)
                K4right = np.random.choice(range(len(right_context_list)), num_context4right, replace=False)
                idx4right += len(K4right)
                right_idx_list.append(idx4right)
                for k in K4right:
                    right_context_batch.append(right_context_list[k])
            else:
                right_idx_list.append(right_idx_list[-1])

            threshold4wrong = min(len(wrong_context_list), self.num_threshold)
            if threshold4wrong > 1:
                num_context4wrong = np.random.randint(1, threshold4wrong)
                K4wrong = np.random.choice(range(len(wrong_context_list)), num_context4wrong, replace=False)
                idx4wrong += len(K4wrong)
                wrong_idx_list.append(idx4wrong)
                for k in K4wrong:
                    wrong_context_batch.append(wrong_context_list[k])
            else:
                wrong_idx_list.append(wrong_idx_list[-1])

        dataset_train = self.objects["mutual_enhance4long_tail"]["dataset_train"]
        device = self.params["device"]
        right_context_batch = context2batch(dataset_train, right_context_batch, device)
        latent_right = self.forward(right_context_batch, kt_model, seq_branch, wright_branch=True)
        wrong_context_batch = context2batch(dataset_train, wrong_context_batch, device)
        latent_wrong = self.forward(wrong_context_batch, kt_model, seq_branch, wright_branch=False)
        question_emb = kt_model.get_target_question_emb(batch_question[0])

        # contextualized representations
        question_emb_transferred = []
        for i in range(len(batch_question[0])):
            mean_right_context = latent_right[right_idx_list[i]: right_idx_list[i+1]]
            mean_wrong_context = latent_wrong[wrong_idx_list[i]: wrong_idx_list[i+1]]
            mean_context = torch.cat((mean_right_context, mean_wrong_context), dim=0)
            question_emb_transferred.append(mean_context.mean(dim=0))

        question_emb_transferred = torch.stack(question_emb_transferred)
        weight_list = torch.FloatTensor(weight_list).view(-1, 1).to(self.params["device"])
        transfer_loss = (weight_list * (question_emb_transferred - question_emb) ** 2).mean()

        return transfer_loss
