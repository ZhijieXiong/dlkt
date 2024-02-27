import torch
import numpy as np

from torch.autograd import Variable


def get_mask4last_or_penultimate(mask_seq, penultimate=False):
    device = mask_seq.device

    with torch.no_grad():
        mask4last = mask_seq.long() - torch.cat((mask_seq[:, 1:], torch.zeros((mask_seq.shape[0], 1)).to(device)), dim=1).long()
        mask4penultimate = mask_seq[:, 1:].long() - torch.cat((mask_seq[:, 2:], torch.zeros((mask_seq.shape[0], 1)).to(device)), dim=1).long()
        mask4penultimate = torch.cat((mask4penultimate, torch.zeros((mask_seq.shape[0], 1)).to(device)), dim=1).long()

    if penultimate:
        return mask4penultimate
    else:
        return mask4last


def parse_question_zero_shot(train_data_statics, question2concept_list, concept2question_list):
    question_zero_shot = train_data_statics["question_zero_fre"]
    question_high_fre = train_data_statics["question_high_fre"]
    question_head4zero = {}

    # 这些zero shot所对应知识点下的head question（出现频率高的）
    for z_q in question_zero_shot:
        concepts_correspond = question2concept_list[z_q]
        qs = []
        for c in concepts_correspond:
            qs += list(set(concept2question_list[c]).intersection(set(question_high_fre)))
        question_head4zero[z_q] = qs

    return question_head4zero


def l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    else:
        d = d.cpu().numpy()
    # "learning_rate": 0.01,"lr_schedule_step": 30,"lr_schedule_gamma": 0.5
    d = d / (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
