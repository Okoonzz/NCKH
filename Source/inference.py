#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference pipeline cho mô hình GCN-xLSTM phát hiện ransomware + Benchmark.

Flow:
1. Nhận vào một (hoặc nhiều) file báo cáo JSON từ Cuckoo.
2. Xây đồ thị (NX DiGraph) → ghi .graphml tạm thời (test.py).
3. Chuyển .graphml → PyG `Data` với embedding DistilBERT (gen_pt.py).
4. Trích tokens từ JSON, ánh xạ sang chỉ mục theo vocab đã huấn luyện.
5. Nạp mô hình đã fine-tune và suy luận xác suất/phân lớp.
6. Benchmark inference: latency, throughput, memory usage.

# Suy luận cho 1 file
python inference.py /path/to/report.json \
    --vocab vocab.json \
    --model best_model_api1000_seq2000.pth

# Hoặc cả thư mục
python inference.py /path/to/folder/with/jsons
"""

import argparse
import json
import os
import tempfile
import time
import psutil

import torch
from torch_geometric.data import Batch

# ---- import các hàm/lớp đã có ----
from build_graphml import build_ransomware_graph, sanitize_and_save_graph
from gen_pt import graphml_to_pyg_data
from multimodal import (
    GCNEncoder,
    xLSTMEncoder,
    MultiModalClassifier,
)

SEQ_LEN = 2000        # cố định như lúc train
API_LIMIT = 1000      # giới hạn chuỗi API lúc trích token


# --------------------------------------------------------------------------
# 1. Hàm trích token giống hệt logic train
# --------------------------------------------------------------------------
def extract_tokens(rep: dict) -> list[str]:
    toks = []

    # 1) API sequence
    for call in rep.get("api_call_sequence", [])[:API_LIMIT]:
        toks.append(f"api:{call.get('api', '')}")

    # 2) Behavior summary
    for ft, vals in rep.get("behavior_summary", {}).items():
        for v in vals:
            toks.append(f"feature:{ft}:{v}")

    # 3) Dropped files
    for d in rep.get("dropped_files", []):
        path = d if not isinstance(d, dict) else d.get("filepath", "")
        toks.append(f"dropped_file:{path}")

    # 4) Signatures
    for sig in rep.get("signatures", []):
        toks.append(f"signature:{sig.get('name', '')}")

    # 5) Processes
    for p in rep.get("processes", []):
        toks.append(f"process:{p.get('name', '')}")

    # 6) Network
    for proto, ents in rep.get("network", {}).items():
        for e in ents:
            if isinstance(e, dict):
                dst = e.get("dst") or e.get("dst_ip", "")
                port = e.get("dst_port") or e.get("port", "")
                toks.append(f"network:{proto}:{dst}:{port}")
            else:
                toks.append(f"network:{proto}:{e}")

    return toks


# --------------------------------------------------------------------------
# 2. Hàm suy luận cho một file JSON
# --------------------------------------------------------------------------
def infer_single(
    json_path: str,
    vocab: dict,
    model: MultiModalClassifier,
    device: torch.device,
) -> tuple[float, int]:
    """
    Trả về (probability, predicted_label), trong đó label 1 = ransomware.
    """
    # --- load báo cáo
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        report = json.load(f)

    # A) Xây PyG Data cho đồ thị
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.splitext(os.path.basename(json_path))[0]
        graphml_path = os.path.join(tmpdir, f"{base}.graphml")

        G = build_ransomware_graph(json_path)
        sanitize_and_save_graph(G, graphml_path)
        data = graphml_to_pyg_data(graphml_path)

    if data is None:
        raise RuntimeError(f"Không tạo được PyG Data cho {json_path}")

    batch_graph = Batch.from_data_list([data]).to(device)

    # B) Chuẩn bị tensor chuỗi token
    tokens = extract_tokens(report)
    idxs = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    if len(idxs) < SEQ_LEN:
        idxs += [vocab["<PAD>"]] * (SEQ_LEN - len(idxs))
    else:
        idxs = idxs[:SEQ_LEN]

    seq_tensor = torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(0)

    # C) Suy luận
    model.eval()
    with torch.no_grad():
        logit = model(batch_graph, seq_tensor)  # (1,)
        prob = torch.sigmoid(logit).item()
        pred = int(prob > 0.5)

    return prob, pred


# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GCN-xLSTM ransomware detector inference.")
    parser.add_argument("input", help="Path tới file JSON hoặc thư mục chứa các JSON cần suy luận")
    parser.add_argument("--vocab", default="vocab.json", help="Đường dẫn tới vocab.json")
    parser.add_argument("--model", default="best_model_api1000_seq2000.pth", help="Đường dẫn tới model .pth đã fine-tune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"], help="Thiết bị suy luận (cpu | cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[INFO] Loaded vocab size = {len(vocab):,}")

    # Khởi tạo model giống lúc train
    graph_enc = GCNEncoder(in_feats=768).to(device)
    seq_enc = xLSTMEncoder(vocab_size=len(vocab), embed=128, seq_len=SEQ_LEN, blocks=1).to(device)
    model = MultiModalClassifier(graph_enc, seq_enc).to(device)

    # Nạp trọng số
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded weights from {args.model}")

    # Lấy danh sách JSON
    if os.path.isdir(args.input):
        json_files = [os.path.join(args.input, fn) for fn in os.listdir(args.input) if fn.endswith(".json")]
    else:
        json_files = [args.input]

    if not json_files:
        raise FileNotFoundError("Không tìm thấy file JSON nào để suy luận!")

    print(f"[INFO] Found {len(json_files)} report(s). Running inference…")

    # Benchmark setup
    process = psutil.Process(os.getpid())
    t0 = time.perf_counter()
    mem_before = process.memory_info().rss / (1024 * 1024)

    results = []
    for jp in sorted(json_files):
        try:
            prob, pred = infer_single(jp, vocab, model, device)
            lbl = "ransomware" if pred == 1 else "benign"
            results.append((jp, prob, lbl))
            print(f"{os.path.basename(jp):<40}  →  prob = {prob:.4f}  |  {lbl}")
        except Exception as e:
            print(f"[ERROR] {jp}: {e}")

    t1 = time.perf_counter()
    mem_after = process.memory_info().rss / (1024 * 1024)

    # Benchmark summary
    n = len(results)
    if n > 0:
        total_time = t1 - t0
        avg_latency = (total_time / n) * 1000  # ms/sample
        throughput = n / total_time            # samples/sec
        mem_used = mem_after - mem_before

        print("\n=== Benchmark Summary ===")
        print(f"Samples processed     : {n}")
        print(f"Total time            : {total_time:.3f} sec")
        print(f"Average latency       : {avg_latency:.2f} ms/sample")
        print(f"Throughput            : {throughput:.2f} samples/sec")
        print(f"Memory before         : {mem_before:.2f} MB")
        print(f"Memory after          : {mem_after:.2f} MB")
        print(f"Memory delta          : {mem_used:.2f} MB")


if __name__ == "__main__":
    main()
