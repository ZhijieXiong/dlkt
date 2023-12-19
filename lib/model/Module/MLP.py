import torch
import torch.nn as nn


class MLP4LLM_emb(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super().__init__()

        self.linear_list = nn.ModuleList([
            nn.Linear(dim_in, 768),
            nn.Linear(768, 256)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(256, dim_out)

    def forward(self, x):
        for lin in self.linear_list:
            x = torch.relu(lin(x))
        return self.out(self.dropout(x))
