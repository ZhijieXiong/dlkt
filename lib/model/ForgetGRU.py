import torch
import torch.nn as nn


class ForgetGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, extra_input_size):
        super(ForgetGRUCell, self).__init__()
        self.hidden_size = hidden_size

        # GRU weights
        self.w_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_in = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hn = nn.Linear(hidden_size, hidden_size, bias=False)

        # Additional weights for the extra input
        self.w_er = nn.Linear(extra_input_size, hidden_size, bias=False)

    def forward(self, x, h, extra_input):
        # Reset gate
        r = torch.sigmoid(self.w_ir(x) + self.w_hr(h) + self.w_er(extra_input))

        # Update gate
        z = torch.sigmoid(self.w_iz(x) + self.w_hz(h))

        # New gate
        n = torch.tanh(self.w_in(x) + r * self.w_hn(h))

        # Hidden state
        h_new = (1 - z) * n + z * h

        return h_new


class ForgetGRU(nn.Module):
    def __init__(self, input_size, hidden_size, extra_input_size, num_layers):
        super(ForgetGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of GRU cells
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(ForgetGRUCell(input_size, hidden_size, extra_input_size))

    def forward(self, x, extra_input):
        batch_size = x.shape[0]
        # Initialize hidden state
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        # Iterate through time steps
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](x_t, h[layer], extra_input)
                x_t = h[layer]
            outputs.append(h[-1].unsqueeze(1))

        return torch.cat(outputs, dim=1), h
