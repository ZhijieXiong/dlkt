import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_latent(z_inferred):
    return torch.randn_like(z_inferred)


class AddEps(nn.Module):
    def __init__(self, params):
        super(AddEps, self).__init__()
        self.params = params

        dim_rnn = self.params["models_config"]["kt_model"]["rnn_layer"]["dim_rnn"]
        self.linear = nn.Sequential(
            nn.Linear(dim_rnn, dim_rnn),
            nn.Tanh()
        )

    def forward(self, x):
        eps = torch.randn_like(x)
        eps = self.linear(eps)

        return eps + x


class FCEncoder(nn.Module):
    def __init__(self, params):
        super(FCEncoder, self).__init__()
        self.params = params

        dim_rnn = self.params["models_config"]["kt_model"]["rnn_layer"]["dim_rnn"]
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.linear1 = nn.Sequential(
            nn.Linear(dim_rnn, dim_rnn),
            nn.Softplus()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(dim_rnn, dim_rnn),
            nn.Softplus()
        )
        self.eps = AddEps(params)
        self.linear_o = nn.Linear(dim_rnn, dim_latent)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = self.eps(self.linear1(x))
        y = self.eps(self.linear2(y))
        y = self.dropout(y)
        y = y + x
        out = self.linear_o(y)

        return out


class FCEncoderNoRes(nn.Module):
    def __init__(self, params):
        super(FCEncoderNoRes, self).__init__()
        self.params = params

        dim_rnn = self.params["models_config"]["kt_model"]["rnn_layer"]["dim_rnn"]
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.linear1 = nn.Sequential(
            nn.Linear(dim_rnn, dim_rnn),
            nn.Softplus()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(dim_rnn, dim_rnn),
            nn.Softplus()
        )
        self.eps = AddEps(params)
        self.linear_o = nn.Linear(dim_rnn, dim_latent)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = self.eps(self.linear1(x))
        y = self.eps(self.linear2(y))
        y = self.dropout(y)

        out = self.linear_o(y)

        return out


class FCEncoderCNN(nn.Module):
    def __init__(self, params):
        super(FCEncoderCNN, self).__init__()
        self.params = params

        dim_rnn = self.params["models_config"]["kt_model"]["rnn_layer"]["dim_rnn"]
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.dot_cnn1 = nn.Sequential(
            nn.Conv1d(dim_rnn, dim_rnn, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.dot_cnn2 = nn.Sequential(
            nn.Conv1d(2 * dim_rnn, dim_rnn, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(dim_rnn, 2 * dim_rnn, kernel_size=5, stride=1, padding=4),
            nn.Softplus()
        )
        self.eps = AddEps(params)
        self.linear_o = nn.Linear(dim_rnn, dim_latent)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        if self.params["models_config"]["kt_model"]["encoder_layer"]["add_eps"]:
            y = self.dot_cnn1(self.eps(x).transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(self.eps(y).transpose(1, 2))[:, :, :-4]
        else:
            y = self.dot_cnn1(x.transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(y.transpose(1, 2))[:, :, :-4]
        y = self.dot_cnn2(y).transpose(1, 2)
        y = x + y

        y = self.dropout(y)
        out = self.linear_o(y)

        return out


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        rnn_config = self.params["models_config"]["kt_model"]["rnn_layer"]
        dim_concept = rnn_config["dim_concept"]
        dim_question = rnn_config["dim_question"]
        dim_correct = rnn_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.linear1 = nn.Linear(dim_latent, dim_emb)
        self.linear2 = nn.Linear(dim_emb, 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, latent):
        latent = self.linear1(latent)
        out_embed = latent

        latent = self.activation(latent)
        predict_score = torch.sigmoid(self.linear2(latent).squeeze(-1))

        return predict_score, out_embed


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        rnn_config = self.params["models_config"]["kt_model"]["rnn_layer"]
        dim_concept = rnn_config["dim_concept"]
        dim_question = rnn_config["dim_question"]
        dim_correct = rnn_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.gru = nn.GRU(dim_emb + dim_latent, 128, batch_first=True)
        self.linear = nn.Linear(128, 1)

    def forward(self, x, z, mask_seq):
        x = F.gelu(self.gru(torch.cat([x, z], dim=-1))[0])
        x = self.linear(x).squeeze(2)

        return mask_seq[:, :-1].float() * x


class AdversaryDiscriminator(nn.Module):
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        rnn_config = self.params["models_config"]["kt_model"]["rnn_layer"]
        dim_concept = rnn_config["dim_concept"]
        dim_question = rnn_config["dim_question"]
        dim_correct = rnn_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.linear_i = nn.Linear(dim_emb + dim_latent, 128)

        self.dnet_list = []
        self.net_list = []
        for _ in range(2):
            self.dnet_list.append(nn.Linear(128, 128))
            self.net_list.append(nn.Linear(128, 128))

        self.dnet_list = nn.ModuleList(self.dnet_list)
        self.net_list = nn.ModuleList(self.net_list)

        self.linear_o = nn.Linear(128, dim_latent)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, z, mask_seq):
        # batch_size x seq_len x dim
        net = torch.cat((x, z), 2)
        net = self.linear_i(net)
        net = self.dropout1(net)

        for i in range(2):
            dnet = self.dnet_list[i](net)
            net = net + self.net_list[i](dnet)
            net = F.elu(net)

        # seq_len
        net = self.linear_o(net)
        net = self.dropout2(net)
        net = net + 0.5 * torch.square(z)

        net = net * mask_seq[:, :-1].float().unsqueeze(2)

        return net
