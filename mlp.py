import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_len):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.seq_len = seq_len
        self.hidden_dim = 1024

        self.embd = nn.Embedding(self.vocab_size, self.embd_dim)
        self.fc1 = nn.Linear(self.embd_dim*self.seq_len, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        B, L = x.shape

        x = self.embd(x) # (B, L, embd_dim)

        x = x.view(B, L*self.embd_dim) # (B, L*embd_dim)

        x = self.fc1(x) # (B, 256)
        x = F.tanh(x) # (B, 256)

        x = self.fc2(x) # (B, vocab_size)

        return x

