import os
import json
import torch


# ================================
# 2. Khai báo các thư mục JSON sau khi unzip
# ================================
benign_json_dirs = [
    "/content/drive/MyDrive/data_benign_extracted/benign",
    "/content/drive/MyDrive/data_benign_extracted1/benign",
]

ransom_json_dirs = [
    "/content/drive/MyDrive/data_ransomware_extracted/ransomware",
    "/content/drive/MyDrive/data_ransomware_extracted1/ransomware",
    "/content/drive/MyDrive/data_ransomware_extracted2025/ransomware"
]

json_dirs = benign_json_dirs + ransom_json_dirs

# File vocab.json sẽ lưu ở đây
VOCAB_JSON  = "/content/drive/MyDrive/vocab_runtime.json"

# Thư mục chứa các file seq_ids .pt
SEQ_OUT_ROOT = "/content/drive/MyDrive/seq_ids"

MAX_LEN = 1500


# ================================
# 3. Hàm tiện ích
# ================================
def iter_feature_files(json_dirs):
    """Duyệt qua tất cả file .json trong list thư mục json_dirs."""
    for jdir in json_dirs:
        if not os.path.isdir(jdir):
            print(f"[WARN] Không tồn tại thư mục JSON: {jdir}")
            continue
        for fname in os.listdir(jdir):
            if fname.endswith(".json"):
                yield os.path.join(jdir, fname)


def build_tokens_from_features(feat):
    """
    Logic build token MultiModalDataset:
    - api_call_sequence (tối đa 1000 API, giữ thứ tự)
    - behavior_summary
    - dropped_files
    - signatures
    - processes
    - network
    """
    toks = []

    # API chuỗi (giữ thứ tự, giới hạn 1000)
    for call in feat.get("api_call_sequence", [])[:1000]:
        toks.append(f"api:{call.get('api','')}")

    # Feature summary
    for ft, vals in feat.get("behavior_summary", {}).items():
        for v in vals:
            toks.append(f"feature:{ft}:{v}")

    # Dropped files
    for d in feat.get("dropped_files", []):
        if isinstance(d, dict):
            toks.append(f"dropped_file:{d.get('filepath','')}")
        else:
            toks.append(f"dropped_file:{str(d)}")

    # Signatures
    for sig in feat.get("signatures", []):
        toks.append(f"signature:{sig.get('name','')}")

    # Processes
    for p in feat.get("processes", []):
        toks.append(f"process:{p.get('name','')}")

    # Network
    for proto, ents in feat.get("network", {}).items():
        for e in ents:
            if isinstance(e, dict):
                dst  = e.get("dst") or e.get("dst_ip", "")
                port = e.get("dst_port") or e.get("port", "")
                toks.append(f"network:{proto}:{dst}:{port}")
            else:
                toks.append(f"network:{proto}:{e}")

    return toks

def main():
    # ================================
    # 4. Build vocab nếu chưa có
    # ================================
    if os.path.isfile(VOCAB_JSON):
        print(f"[INFO] Đã có vocab: {VOCAB_JSON} → load.")
        with open(VOCAB_JSON, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    else:
        print(f"[INFO] CHƯA có vocab → build mới từ tất cả JSON.")
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        next_idx = 2

        count_files = 0
        for fpath in iter_feature_files(json_dirs):
            count_files += 1
            if count_files % 100 == 0:
                print(f"  [VOCAB] Đang quét file thứ {count_files}: {fpath}")

            with open(fpath, "r", encoding="utf-8") as f:
                feat = json.load(f)

            toks = build_tokens_from_features(feat)
            for t in toks:
                if t not in vocab:
                    vocab[t] = next_idx
                    next_idx += 1

        os.makedirs(os.path.dirname(VOCAB_JSON), exist_ok=True)
        with open(VOCAB_JSON, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Đã build vocab xong, tổng số token = {len(vocab)}")
        print(f"[INFO] Đã lưu vocab vào: {VOCAB_JSON}")


    # ================================
    # 5. Encode từng JSON → tensor ID [1500] và lưu .pt
    # ================================
    os.makedirs(SEQ_OUT_ROOT, exist_ok=True)

    total = 0
    for fpath in iter_feature_files(json_dirs):
        total += 1
        if total % 100 == 0:
            print(f"[SEQ] Đang xử lý file thứ {total}: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            feat = json.load(f)

        toks = build_tokens_from_features(feat)
        toks = toks[:MAX_LEN]  # Cắt 1500

        ids = [vocab.get(t, vocab["<UNK>"]) for t in toks]
        if len(ids) < MAX_LEN:
            ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

        # Xác định class để lưu ra thư mục benign / ransomware
        if "/benign/" in fpath:
            cls = "benign"
        elif "/ransomware/" in fpath:
            cls = "ransomware"
        else:
            cls = "unknown"

        out_dir = os.path.join(SEQ_OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        sid = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(out_dir, f"{sid}_seq.pt")

        torch.save(torch.tensor(ids, dtype=torch.long), out_path)

    print(f"[DONE] Đã encode xong {total} file JSON.")
    print(f"[DONE] Các file .pt nằm trong: {SEQ_OUT_ROOT}")
    print(f"[DONE] vocab.json nằm ở: {VOCAB_JSON}")
    pass

if __name__ == "__main__":
    main()