import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_model_summary import summary


class LSTM(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_len):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.seq_len = seq_len
        self.hidden_size = 1024

        self.embd = nn.Embedding(self.vocab_size, self.embd_dim)
        self.lstm = nn.LSTM(
                input_size=self.embd_dim, 
                hidden_size=self.hidden_size, 
                num_layers=2,
                batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        B, L = x.shape

        x = self.embd(x) # B, L, embd_dim

        x, h_n = self.lstm(x) # B, L, hidden_size
        assert x.shape == (B, L, self.hidden_size)
        x = x[:, -1, :] # B, hidden_size
        assert x.shape == (B, self.hidden_size)

        x = self.fc(x) # B, vocab_size

        return x


if __name__ == '__main__':
    BATCH_SIZE = 1
    SEQ_LEN = 256
    x = torch.randint(0, 111, size=(BATCH_SIZE, SEQ_LEN))
    model = LSTM(111, 32, SEQ_LEN)
    print(summary(model, x))
