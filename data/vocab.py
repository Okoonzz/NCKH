# data/vocab.py

from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(sequences, max_size=100000, min_freq=2):
    """
    Xây vocab từ list các sequence (list of tokens).
    - max_size: số token nhiều nhất (bao gồm PAD và UNK).
    - min_freq: chỉ giữ token có tần suất >= min_freq.
    Luồng:
      1) Đếm toàn bộ token qua Counter.
      2) Lọc token có freq < min_freq.
      3) Lấy top max_size-2 token theo freq giảm dần.
      4) Gán index 0->PAD, 1->UNK, sau đó lần lượt các token còn lại.
    """
    # Đếm tần suất tất cả token
    counter = Counter(tok for seq in sequences for tok in seq)
    # Lọc theo min_freq
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
    # Sắp theo freq giảm dần
    items.sort(key=lambda x: x[1], reverse=True)
    # Giới hạn max_size-2
    top_tokens = [tok for tok, _ in items[: max_size - 2]]

    # Tạo vocab
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, tok in enumerate(top_tokens, start=2):
        vocab[tok] = idx

    return vocab

def encode_sequence(seq, vocab, max_len):
    """
    Chuyển list token thành list index, pad/truncate về độ dài max_len.
    - Tokens không trong vocab sẽ được map về UNK (index 1).
    - PAD_TOKEN (index 0) dùng để pad.
    """
    unk_idx = vocab.get(UNK_TOKEN, 1)
    pad_idx = vocab.get(PAD_TOKEN, 0)

    # Map token -> index
    idxs = [vocab.get(tok, unk_idx) for tok in seq]
    # Truncate
    if len(idxs) > max_len:
        idxs = idxs[:max_len]
    # Pad
    elif len(idxs) < max_len:
        idxs += [pad_idx] * (max_len - len(idxs))
    return idxs
