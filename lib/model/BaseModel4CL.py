import torch
import torch.nn as nn
import numpy as np
from abc import *

from .loss_util import duo_info_nce, meta_contrast_rl
from .util import get_mask4last_or_penultimate


class BaseModel4CL:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    @abstractmethod
    def get_latent(self, batch):
        pass

    @abstractmethod
    def get_latent_last(self, batch):
        pass

    @abstractmethod
    def get_latent_mean(self, batch):
        pass

    @abstractmethod
    def get_latent_last_from_adv_data(self, dataset, batch):
        pass

    @abstractmethod
    def get_latent_mean_from_adv_data(self, dataset, batch):
        pass

    def get_duo_cl_loss(self, batch, latent_type):
        batch_aug = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_hard_neg = {
            "concept_seq": batch["concept_seq_hard_neg"],
            "question_seq": batch["question_seq_hard_neg"],
            "correct_seq": batch["correct_seq_hard_neg"],
            "mask_seq": batch["mask_seq_hard_neg"]
        }

        if latent_type == "last_time":
            latent_ori_pooled = self.get_latent_last(batch)
            latent_aug_pooled = self.get_latent_last(batch_aug)
            latent_hard_neg_pooled = self.get_latent_last(batch_hard_neg)
        elif latent_type == "mean_pool":
            latent_ori_pooled = self.get_latent_mean(batch)
            latent_aug_pooled = self.get_latent_mean(batch_aug)
            latent_hard_neg_pooled = self.get_latent_mean(batch_hard_neg)
        else:
            raise NotImplementedError()

        temp = self.params["other"]["duo_cl"]["temp"]
        cl_loss = duo_info_nce(latent_ori_pooled, latent_aug_pooled, temp,
                               sim_type="cos", z_hard_neg=latent_hard_neg_pooled)

        return cl_loss

    def get_instance_cl_loss(self, batch, latent_type, use_random_seq_len=False, use_adv_aug=False, dataset=None):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        if latent_type == "last_time" and not use_adv_aug:
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
        elif latent_type == "mean_pool" and not use_adv_aug:
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
        elif latent_type == "last_time" and use_adv_aug:
            latent_aug0_pooled = self.get_latent_last_from_adv_data(dataset, batch_aug0)
            latent_aug1_pooled = self.get_latent_last_from_adv_data(dataset, batch_aug1)
        elif latent_type == "mean_pool" and use_adv_aug:
            latent_aug0_pooled = self.get_latent_mean_from_adv_data(dataset, batch_aug0)
            latent_aug1_pooled = self.get_latent_mean_from_adv_data(dataset, batch_aug1)
        else:
            raise NotImplementedError()

        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim_aug = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug1_pooled.unsqueeze(0),
                                              dim=-1) / temp

        if "correct_seq_hard_neg" in batch.keys():
            if use_random_seq_len:
                batch_hard_neg = {
                    "concept_seq": batch["concept_seq_random_len"],
                    "question_seq": batch["question_seq_random_len"],
                    "correct_seq": batch["correct_seq_random_len"],
                    "mask_seq": batch["mask_seq_random_len"]
                }
            else:
                batch_hard_neg = {
                    "concept_seq": batch["concept_seq"],
                    "question_seq": batch["question_seq"],
                    "correct_seq": batch["correct_seq_hard_neg"],
                    "mask_seq": batch["mask_seq"]
                }
            if latent_type == "last_time" and not use_adv_aug:
                latent_hard_neg_pooled = self.get_latent_last(batch_hard_neg)
            elif latent_type == "mean_pool" and not use_adv_aug:
                latent_hard_neg_pooled = self.get_latent_mean(batch_hard_neg)
            elif latent_type == "last_time" and use_adv_aug:
                latent_hard_neg_pooled = self.get_latent_last_from_adv_data(dataset, batch_hard_neg)
            elif latent_type == "mean_pool" and use_adv_aug:
                latent_hard_neg_pooled = self.get_latent_mean_from_adv_data(dataset, batch_hard_neg)
            else:
                raise NotImplementedError()
            cos_sim_neg = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1),
                                                  latent_hard_neg_pooled.unsqueeze(0), dim=-1) / temp
            cos_sim = torch.cat((cos_sim_aug, cos_sim_neg), dim=1)
        else:
            cos_sim = cos_sim_aug

        labels = torch.arange(cos_sim.size(0)).long().to(self.params["device"])
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_instance_cl_loss_all_interaction(self, batch, use_adv_aug=False, dataset=None):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        latent_aug0 = self.get_latent(batch_aug0)
        mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)
        latent_aug0_last = latent_aug0[torch.where(mask4last_aug0 == 1)]

        latent_aug1 = self.get_latent(batch_aug1)
        mask4last_aug1 = get_mask4last_or_penultimate(batch_aug1["mask_seq"], penultimate=False)
        latent_aug1_last = latent_aug1[torch.where(mask4last_aug1 == 1)]

        bs = latent_aug0.shape[0]
        seq_len = latent_aug0.shape[1]
        m = (torch.eye(bs) == 0)

        # 将另一增强序列的每个时刻都作为一个neg
        neg_all = latent_aug1.repeat(bs, 1, 1).reshape(bs, bs, seq_len, -1)[m].reshape(bs, bs - 1, seq_len, -1)
        mask_bool4neg = torch.ne(batch["mask_seq_aug_1"].repeat(bs, 1).reshape(bs, bs, -1)[m].reshape(bs, bs - 1, -1),
                                 0)

        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim_list = []
        for i in range(bs):
            anchor = latent_aug0_last[i]
            pos = latent_aug1_last[i]
            neg = neg_all[i][:, 1:][mask_bool4neg[i][:, 1:]]
            sim_i = torch.cosine_similarity(anchor, torch.cat((pos.unsqueeze(dim=0), neg), dim=0)) / temp
            cos_sim_list.append(sim_i.unsqueeze(dim=0))

        labels = torch.tensor([0]).long().to(self.params["device"])
        cl_loss = 0.
        for i in range(bs):
            cos_sim = cos_sim_list[i]
            cl_loss = cl_loss + nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_cluster_cl_loss(self, batch, clus, latent_type, use_random_seq_len=False, use_adv_aug=False, dataset=None):
        if use_random_seq_len:
            batch_ori = {
                "concept_seq": batch["concept_seq_random_len"],
                "question_seq": batch["question_seq_random_len"],
                "correct_seq": batch["correct_seq_random_len"],
                "mask_seq": batch["mask_seq_random_len"]
            }
        else:
            batch_ori = batch
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }
        batch_size = batch["mask_seq"].shape[0]

        if latent_type == "last_time" and not use_adv_aug:
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
            latent_ori_pooled = self.get_latent_last(batch_ori)
        elif latent_type == "mean_pool" and not use_adv_aug:
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
            latent_ori_pooled = self.get_latent_mean(batch_ori)
        elif latent_type == "last_time" and use_adv_aug:
            latent_aug0_pooled = self.get_latent_last_from_adv_data(dataset, batch_aug0)
            latent_aug1_pooled = self.get_latent_last_from_adv_data(dataset, batch_aug1)
            latent_ori_pooled = self.get_latent_last_from_adv_data(dataset, batch_ori)
        elif latent_type == "mean_pool" and use_adv_aug:
            latent_aug0_pooled = self.get_latent_mean_from_adv_data(dataset, batch_aug0)
            latent_aug1_pooled = self.get_latent_mean_from_adv_data(dataset, batch_aug1)
            latent_ori_pooled = self.get_latent_mean_from_adv_data(dataset, batch_ori)
        else:
            raise NotImplementedError()
        state = np.array(latent_ori_pooled.detach().cpu().tolist())
        intent_id, intent = clus.query(state)

        intent_id0 = intent_id.contiguous().view(-1, 1)
        intent_id1 = intent_id.contiguous().view(1, -1)
        mask4inf = (intent_id0 == intent_id1) & torch.ne(torch.eye(batch_size), 1).to(self.params["device"])

        temp = self.params["other"]["cluster_cl"]["temp"]
        cos_sim_aug0 = torch.cosine_similarity(intent.unsqueeze(1), latent_aug0_pooled.unsqueeze(0), dim=-1) / temp
        cos_sim_aug1 = torch.cosine_similarity(intent.unsqueeze(1), latent_aug1_pooled.unsqueeze(0), dim=-1) / temp
        cos_sim_aug0[mask4inf] = -(1 / temp)
        cos_sim_aug1[mask4inf] = -(1 / temp)

        labels = torch.arange(batch_size).long().to(self.params["device"])
        cl_loss0 = nn.functional.cross_entropy(cos_sim_aug0, labels)
        cl_loss1 = nn.functional.cross_entropy(cos_sim_aug1, labels)

        return (cl_loss0 + cl_loss1) / 2

    def get_meta_contrast_cl_loss(self, batch, latent_type, meta_extractors, use_regularization, use_adv_aug=False, dataset=None):
        batch_size = batch["mask_seq"].shape[0]
        extractor0, extractor1 = meta_extractors

        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        if latent_type == "last_time":
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
        elif latent_type == "mean_pool":
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
        else:
            raise NotImplementedError()
        latent_aug0_extracted = extractor0(latent_aug0_pooled)
        latent_aug1_extracted = extractor1(latent_aug1_pooled)

        temp = self.params["other"]["instance_cl"]["temp"]
        labels = torch.arange(batch_size).long().to(self.params["device"])
        # 随机增强的对比损失
        cos_sim_0 = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug1_pooled.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_0 = nn.functional.cross_entropy(cos_sim_0, labels)

        # 计算meta cl loss
        cos_sim_1 = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug0_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_1 = nn.functional.cross_entropy(cos_sim_1, labels)

        cos_sim_2 = torch.cosine_similarity(latent_aug1_pooled.unsqueeze(1), latent_aug1_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_2 = nn.functional.cross_entropy(cos_sim_2, labels)

        cos_sim_3 = torch.cosine_similarity(latent_aug0_extracted.unsqueeze(1), latent_aug1_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_3 = nn.functional.cross_entropy(cos_sim_3, labels)

        if use_regularization:
            rl_loss = 0.
            rl_loss += meta_contrast_rl(latent_aug0_pooled, latent_aug1_extracted, temp, "dot")
            rl_loss += meta_contrast_rl(latent_aug1_pooled, latent_aug0_extracted, temp, "dot")
            rl_loss += meta_contrast_rl(latent_aug0_extracted, latent_aug1_extracted, temp, "dot")
        else:
            rl_loss = None

        return cl_loss_0, cl_loss_1 + cl_loss_2 + cl_loss_3, rl_loss