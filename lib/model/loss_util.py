import torch
import torch.nn as nn


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def duo_info_nce(z_i, z_j, temp, sim_type="cos", z_hard_neg=None):
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

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = nn.functional.cross_entropy(logits, labels)

    return loss
