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
    out = -1.0 * (p * torch.log(p + 1e-8) + (1 - p) * torch.log((1 - p + 1e-8)))
    return out.mean()


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N)).bool()
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def meta_contrast(z_i, z_j, temp, sim_type):
    batch_size = z_i.shape[0]
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    if sim_type == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim_type == 'dot':
        sim = torch.mm(z, z.T) / temp
    else:
        raise NotImplementedError()
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)

    return positive_samples, negative_samples


def meta_contrast_rl(a, b, temp, sim_type):
    ori_p, ori_n = meta_contrast(a, b, temp, sim_type)
    min_positive_value, min_pos_pos = torch.min(ori_p, dim=-1)
    max_negative_value, max_neg_pos = torch.max(ori_n, dim=-1)
    lgamma_margin_pos, _ = torch.min(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                .unsqueeze(1)), dim=1), dim=-1)
    lgamma_margin_pos = lgamma_margin_pos.unsqueeze(1)
    lgamma_margin_neg, _ = torch.max(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                .unsqueeze(1)), dim=1), dim=-1)
    lgamma_margin_neg = lgamma_margin_neg.unsqueeze(1)
    loss = torch.mean(torch.clamp(ori_p - lgamma_margin_pos, min=0))
    loss += torch.mean(torch.clamp(lgamma_margin_neg - ori_n, min=0))

    return loss
