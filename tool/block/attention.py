import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LuongAttention', 'BahdanauAttention']


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.concat_fc = nn.Linear(hidden_size * 2, hidden_size)

    def energy(self, hidden, hidden_sequence):
        concat_hidden = torch.cat(
            (hidden.expand(hidden_sequence.size(0), -1, -1), hidden_sequence),
            dim=2)
        energy = torch.tanh(self.attention(concat_hidden))
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, hidden_sequence, input_masks):
        energy = self.energy(hidden, hidden_sequence)
        energy.masked_fill_(input_masks.logical_not(),
                            value=torch.tensor(-1e10,
                                               dtype=torch.float)).permute(
                                                   1, 0)
        attention_weight = F.softmax(energy,
                                     dim=1).unsqueeze(dim=1).permute(2, 1, 0)
        context = attention_weight.bmm(hidden_sequence.permute(1, 0,
                                                               2)).permute(
                                                                   1, 0, 2)
        return context


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def energy(self, hidden, hidden_sequence):
        hidden = hidden.expand(hidden_sequence.size(0), -1, -1)
        energy = self.w1(hidden_sequence) + self.w2(hidden)
        energy = torch.sum(self.v * torch.tanh(energy), dim=2)
        return energy

    def forward(self, hidden, hidden_sequence, input_masks):
        energy = self.energy(hidden, hidden_sequence)
        energy.masked_fill_(input_masks.logical_not(),
                            value=torch.tensor(-1e10,
                                               dtype=torch.float)).permute(
                                                   1, 0)
        attention_weight = F.softmax(energy, dim=1).unsqueeze(1)
        context = attention_weight.bmm(hidden_sequence.permute(1, 0,
                                                               2)).permute(
                                                                   1, 0, 2)
        return context
