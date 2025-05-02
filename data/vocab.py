def build_vocab(event_sequences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for seq in event_sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len):
    unk = vocab.get("<UNK>", 1)
    indices = [vocab.get(tok, unk) for tok in seq]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        pad = vocab.get("<PAD>", 0)
        indices += [pad] * (max_len - len(indices))
    return indices