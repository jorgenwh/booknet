import torch

from mlp import MLP
from rnn import RNN
from lstm import LSTM
from transformer import Transformer
from helpers import read_text, preprocess_text, AverageMeter


data, vocab_size, char_to_index, index_to_char = preprocess_text(read_text("data/tinyshakespeare.txt"))
assert data.dtype == torch.int64


BATCH_SIZE = 8
LEARNING_RATE = 0.001
START_EPOCH = 0
EPOCHS = 5
EMBD_DIM = 32
SEQ_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch():
    X = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.int64)
    y = torch.zeros((BATCH_SIZE), dtype=torch.int64)
    indices = torch.randint(0, data.size(0) - SEQ_LEN, (BATCH_SIZE,))
    
    for i in range(BATCH_SIZE):
        start_index = indices[i]
        end_index = start_index + SEQ_LEN
        X[i, :] = data[start_index:end_index]
        y[i] = data[end_index]

    return X, y


model = LSTM(
        vocab_size, 
        EMBD_DIM, 
        SEQ_LEN
)
model = Transformer(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size, 
        max_len=SEQ_LEN, 
        d_model=EMBD_DIM,
        d_ff=EMBD_DIM*4,
        n_heads=4,
        n_layers=2,
        dropout=0.1
)

if START_EPOCH > 0:
    model.load_state_dict(torch.load("models/model_epoch" + str(START_EPOCH) + ".pth"))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

iters = data.size(0) // BATCH_SIZE
min_loss = float("inf")

for epoch in range(START_EPOCH, EPOCHS):
    # adjust learning rate
    lr = LEARNING_RATE * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("Epoch: " + str(epoch+1) + "/" + str(EPOCHS))

    iloss = AverageMeter()

    model.train()

    for i in range(iters):
        X, y = get_batch()
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # forward pass
        output = model(X)

        # compute loss
        loss = loss_fn(output, y)
        iloss.update(loss.item(), X.size(0))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "batch: " + str(i + 1) + "/" + str(iters) + " | " +
            "loss: " + str(iloss) + " "*10, end="\r"
        )

    print(
        "batch: " + str(iters) + "/" + str(iters) + " | " +
        "loss: " + str(iloss) + " "*10
    )

    if iloss.avg < min_loss:
        min_loss = iloss.avg
        torch.save(model.state_dict(), "models/model_epoch" + str(epoch + 1) + ".pth")
        print("saved models/model_epoch" + str(epoch + 1) + ".pth")

