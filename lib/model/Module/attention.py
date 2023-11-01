import torch
import torch.nn.functional as F
from torch.nn import Softplus


def attention_AKT(q, k, v, dim_head, mask, dropout, zero_pad, gamma=None, pdiff=None, device="cpu"):
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.tensor(dim_head).float().sqrt().to(device)
    batch_size, num_head, seq_len = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seq_len).expand(seq_len, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        # batch_size, num_head, seq_len, seq_len
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distance_cumulative = torch.cumsum(scores_, dim=-1)
        distance_total = torch.sum(scores_, dim=-1, keepdim=True)

        # 1, 1, seq_len, seq_len 位置差值
        position_effect = torch.abs(x1 - x2)[None, None, :, :]
        # score <0 时，设置为0
        dist_scores = torch.clamp((distance_total - distance_cumulative) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = Softplus()
    # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    gamma = -1. * m(gamma).unsqueeze(0)

    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff is None:
        # 对应论文公式1中的新增部分
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        # 对应论文公式1中的新增部分
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    # batch_size, num_head, seq_len, seq_len
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(batch_size, num_head, 1, seq_len).to(device)
        # 第一行score置0
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output
