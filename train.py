import torch


from helpers import read_text, preprocess_text


data, vocab_size, char_to_index, index_to_char = preprocess_text(read_text("data/tinyshakespeare.txt"))
assert data.dtype == torch.int64
