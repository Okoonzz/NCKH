#!/usr/bin/env python3
import os
import glob
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from json import JSONDecodeError

# 1) Import các hàm xử lý data và model
from data.sequence_vector import load_report, build_text_event_sequence
from data.vocab import build_vocab, encode_sequence
from model.xlstm_classifier import xLSTMClassifier

# --- CẤU HÌNH CHUNG ---
BENIGN_DIR   = "reports/benign"
RANSOM_DIR   = "reports/ransomware"
VOCAB_PATH   = "vocab.json"
BEST_MODEL   = "model_best_2.pt"
MAX_SEQ_LEN  = 50       # phải khớp với inference.py
EMBED_DIM    = 128
BATCH_SIZE   = 16
LR           = 1e-3
EPOCHS       = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

def prepare_splits():
    # Load paths + labels
    benign = glob.glob(os.path.join(BENIGN_DIR, "*.json"))
    ransom = glob.glob(os.path.join(RANSOM_DIR, "*.json"))
    paths  = benign + ransom
    labels = [0]*len(benign) + [1]*len(ransom)

    # 80% train+val, 20% test
    p_trainval, p_test, y_trainval, y_test = train_test_split(
        paths, labels,
        test_size=0.20,
        stratify=labels,
        random_state=SEED
    )
    # Trong train+val, lấy 12.5% làm val (0.125*0.8=0.1 tổng)
    p_train, p_val, y_train, y_val = train_test_split(
        p_trainval, y_trainval,
        test_size=0.125,
        stratify=y_trainval,
        random_state=SEED
    )

    print(f"Splits: train={len(p_train)}, val={len(p_val)}, test={len(p_test)}")
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)

def build_dataset(paths, labels, vocab):
    seqs, ys = [], []
    for p, y in zip(paths, labels):
        try:
            rpt = load_report(p)
        except JSONDecodeError:
            print(f"[Warning] skip invalid JSON: {p}")
            continue
        seqs.append(build_text_event_sequence(rpt))
        ys.append(y)

    X = [encode_sequence(s, vocab, MAX_SEQ_LEN) for s in seqs]
    X_t = torch.tensor(X, dtype=torch.long)
    y_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X_t, y_t)

def evaluate_on_loader(model, loader):
    """Chạy inference trên loader, trả về y_true và y_pred."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().squeeze(1)
            preds  = (probs >= 0.5).long().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().squeeze(1).long().tolist())
    return y_true, y_pred

def main():
    torch.manual_seed(SEED)

    # -------- 1) Chuẩn bị train/val/test splits --------
    (p_train, y_train), (p_val, y_val), (p_test, y_test) = prepare_splits()

    # -------- 2) Xây vocab từ train+val --------
    seqs_tv = []
    for p in p_train + p_val:
        rpt = load_report(p)
        seqs_tv.append(build_text_event_sequence(rpt))
    vocab = build_vocab(seqs_tv)
    with open(VOCAB_PATH, "w", encoding="utf-8", errors="replace") as f:
        json.dump(vocab, f, ensure_ascii=True, indent=2)
    print(f"Vocab size = {len(vocab)} (saved to {VOCAB_PATH})")

    # -------- 3) Tạo DataLoader cho từng split --------
    ds_train = build_dataset(p_train, y_train, vocab)
    ds_val   = build_dataset(p_val,   y_val,   vocab)
    ds_test  = build_dataset(p_test,  y_test,  vocab)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"DataLoaders → train {len(ds_train)}, val {len(ds_val)}, test {len(ds_test)}")

    # -------- 4) Khởi tạo model, loss, optimizer --------
    model   = xLSTMClassifier(vocab_size=len(vocab),
                              embed_dim=EMBED_DIM,
                              context_length=MAX_SEQ_LEN).to(DEVICE)
    loss_fn = BCEWithLogitsLoss()
    optim   = Adam(model.parameters(), lr=LR)
    print(f"Model on {DEVICE}. Training {EPOCHS} epochs...")

    # -------- 5) Train + Validation --------
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        total_tr_loss = 0.0
        for xb, yb in loader_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_tr_loss += loss.item() * xb.size(0)
        avg_tr_loss = total_tr_loss / len(ds_train)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss   = loss_fn(logits, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(ds_val)

        print(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_tr_loss:.4f}  val_loss={avg_val_loss:.4f}")
        # Lưu best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "max_seq_len": MAX_SEQ_LEN,
                "embed_dim": EMBED_DIM
            }, BEST_MODEL)
            print(f"  → Best model saved (val_loss={best_val_loss:.4f})")

    print("Training & validation complete.")
    print(f"Best model checkpoint: {BEST_MODEL}")

    # -------- 6) Test final evaluation --------
    # Load best model
    ckpt = torch.load(BEST_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    # Đánh giá trên test set
    y_true, y_pred = evaluate_on_loader(model, loader_test)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    print("\nTest Set Evaluation:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

if __name__ == "__main__":
    main()
