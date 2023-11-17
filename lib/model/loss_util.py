import torch
import torch.nn as nn


def trans2neg(one_view):
    bs = one_view.shape[0]
    neg_view = one_view.repeat(1, bs, 1).reshape(bs, bs, -1)
    m = (torch.eye(bs) == 0)
    return neg_view[m].reshape(bs, bs-1, -1)


def duo_info_nce(z_i, z_j, temp, sim_type="cos", z_hard_neg=None):
    batch_size = z_i.shape[0]
    device = z_i.device

    if sim_type == 'cos':
        sim = nn.functional.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) / temp
    else:
        raise NotImplementedError()

    if z_hard_neg is not None:
        neg_sim = nn.functional.cosine_similarity(z_i, z_hard_neg, dim=1) / temp
        sim = torch.cat((sim, neg_sim.unsqueeze(1)), dim=1)

    labels = torch.arange(batch_size).long().to(device)
    loss = nn.functional.cross_entropy(sim, labels)

    return loss


def binary_entropy(p):
    out = -1.0 * (p * torch.log(p) + (1 - p) * torch.log((1 - p)))
    return out.mean()
