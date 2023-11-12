import torch


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
