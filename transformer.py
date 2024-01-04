import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len=128, d_model=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.input_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = torch.zeros((seq_len, d_model), dtype=torch.float32)
        for k in range(seq_len):
            for j in range(d_model):
                if j % 2 == 0:
                    self.positional_encoding[k, j] = torch.sin(k/10000 ** (2*j / d_model))
                else:
                    self.positional_encoding[k, j] = torch.cos(k/10000 ** (2*j / d_model))
