import torch


def read_text(filename):
    with open(filename, "r") as f:
        return f.read()

def preprocess_text(text):
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}
    data = torch.tensor([char_to_index[char] for char in text])

    return data, vocab_size, char_to_index, index_to_char

if __name__ == "__main__":
    text = read_data()
    print(len(text))
    data, vocab_size, char_to_index, index_to_char = preprocess_text(text)
    print(data)
    print(data.dtype)
    print(data.shape)

    print(char_to_index)

    print(text[:10])
    print(data[:10])
