import argparse
import time
import torch

from mlp import MLP
from rnn import RNN
from lstm import LSTM
from helpers import read_text, preprocess_text, AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="Number of new tokens to generate", default=500)
parser.add_argument("-m", type=str, help="Model name", default=None)
args = parser.parse_args()

data, vocab_size, char_to_index, index_to_char = preprocess_text(read_text("data/tinyshakespeare.txt"))

MAX_NEW_TOKENS = args.t
EMBD_DIM = 32
SEQ_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#model = MLP(vocab_size, EMBD_DIM, SEQ_LEN)
#model = RNN(vocab_size, EMBD_DIM, SEQ_LEN)
model = LSTM(vocab_size, EMBD_DIM, SEQ_LEN)
#model.load_state_dict(torch.load("models/model_epoch1.pth"))
if args.m is not None:
    model.load_state_dict(torch.load(args.m))
model = model.to(DEVICE)

prompt = torch.zeros(1, SEQ_LEN, dtype=torch.long)
str_prompt = input("PROMPT: ")
for i, char in enumerate(str_prompt):
    prompt[0, SEQ_LEN - len(str_prompt) + i] = char_to_index[char]
prompt = prompt.to(DEVICE)

model.eval()
with torch.no_grad():
    for i in range(MAX_NEW_TOKENS):
        output = model(prompt)

        new_token = torch.argmax(output, dim=-1)
        str_token = index_to_char[new_token.item()]

        prompt = prompt[:, 1:]
        new_token = new_token.reshape(1, 1)
        prompt = torch.cat((prompt, new_token), dim=1)

        print(str_token, end="", flush=True)
        time.sleep(0.01)

print()
