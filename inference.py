#!/usr/bin/env python3
import os
import sys
import json
from json.decoder import JSONDecodeError

import torch

# 1) Import các hàm xử lý data và model
from data.sequence_vector import load_report, build_text_event_sequence
from data.vocab import encode_sequence
from model.xlstm_classifier import xLSTMClassifier

# --- CẤU HÌNH ---
VOCAB_PATH   = "vocab.json"
BEST_MODEL   = "model_best_2.pt"
FALLBACK_MODEL = "model.pt"
MAX_SEQ_LEN  = 50     # phải khớp với train.py
EMBED_DIM    = 128    # phải khớp với train.py

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/.json")
        sys.exit(1)

    features_path = sys.argv[1]
    # 2) Load vocab
    if not os.path.exists(VOCAB_PATH):
        print(f"[Error] Cannot find vocab file: {VOCAB_PATH}")
        sys.exit(1)
    with open(VOCAB_PATH, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    vocab_size = len(vocab)

    # 3) Khởi tạo model & load checkpoint
    model = xLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        context_length=MAX_SEQ_LEN
    )
    # Load best checkpoint nếu có, ngược lại load fallback
    ckpt_path = BEST_MODEL if os.path.exists(BEST_MODEL) else FALLBACK_MODEL
    if not os.path.exists(ckpt_path):
        print(f"[Error] Cannot find model checkpoint: {BEST_MODEL} or {FALLBACK_MODEL}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Nếu checkpoint lưu dict chứa "model_state"
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 4) Load features JSON và build text sequence
    try:
        features = load_report(features_path)
    except JSONDecodeError:
        print(f"[Error] Invalid JSON file: {features_path}")
        sys.exit(1)

    seq = build_text_event_sequence(features)
    if not seq:
        print("[Warning] Generated empty event sequence.")
    # 5) Encode & pad sequence
    idxs = encode_sequence(seq, vocab, MAX_SEQ_LEN)
    xb   = torch.tensor([idxs], dtype=torch.long)

    # 6) Predict
    with torch.no_grad():
        logits = model(xb)
        prob   = torch.sigmoid(logits).item()
    label = "RANSOMWARE" if prob >= 0.5 else "BENIGN"

    # 7) Output result
    print(f"Features file: {features_path}")
    print(f"→ Probability ransomware: {prob:.4f}")
    print(f"→ Predicted label      : {label}")

if __name__ == "__main__":
    main()
