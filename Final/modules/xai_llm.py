# ============================================================
# XAI for CURRENT model (GCN + xLSTM + Gating w_seq/w_graph)
# - LIME for sequence branch (from seq_ids -> token strings)
# - GNNExplainer for graph branch (node_ids -> token names)
# - LLM (OpenAI API) to turn XAI output into Markdown report
# ============================================================

import os, re, json, csv
import numpy as np
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

# PyG explainer (new API)
try:
    from torch_geometric.explain import Explainer
    from torch_geometric.explain.algorithm import GNNExplainer
except Exception:
    from torch_geometric.explain import Explainer, GNNExplainer

from lime.lime_text import LimeTextExplainer

# xLSTM
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig
)

# OpenAI client (optional)
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# ======================================================
# CONFIG
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Graph paths
BENIGN_GRAPH_DIR = "/kaggle/input/ptbenign-for-xai/PYG_benign_rans21/benign"
RANSOMWARE_GRAPH_DIRS = [
    "/kaggle/input/ptclean-for-xai/pyg_data_DistilBERT_ransomware_XAI_2025/ransomware",
    "/kaggle/input/ptnoise-for-xai/pyg_data_DistilBERT_ransomware_XAI_noise_2025/ransomware"
]

# Sequence paths
SEQ_ROOT = "/kaggle/input/seq-ids/seq_ids"


SEQ_VOCAB_JSON = os.getenv("SEQ_VOCAB_JSON", "artifacts/seq_vocab.json")
GRAPH_ID2TOKEN_JSON = os.getenv("GRAPH_ID2TOKEN_JSON", "artifacts/graph_vocab_id2token.json")
GRAPH_VOCAB_JSON = os.getenv("GRAPH_VOCAB_JSON", "artifacts/graph_vocab.json")
MODEL_PT = os.getenv("MODEL_PT", "artifacts/best_mm_da_seq1500.pt")

# Output
SAVE_PREFIX = "/kaggle/working/xai_single_sample"

# Choose sample
INDEX = 555

# LIME params
LIME_TOP_K = 20
LIME_NUM_SAMPLES = 600
LIME_BATCH_SIZE = 16
LIME_MAX_TOKENS_FOR_TEXT = 500  # decode/tạo text cho LIME từ N token đầu

# GNNExplainer params
GNN_EPOCHS = 120
GNN_TOP_RATIO = 0.10  # % edges/nodes quan trọng

# ======================================================
# Helpers: naming + vocab loaders
# ======================================================

def normalize_name(name: str) -> str:
    name = name.replace("_seq.pt", "")
    name = name.replace(".pt", "")
    name = re.sub(r"\.(e?xe?)$", "", name, flags=re.IGNORECASE)
    return name

def load_seq_vocab(vocab_path: str):
    """
    Expect vocab_runtime.json is token->id (as in training).
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok2id = json.load(f)

    # robust pad/unk
    pad_id = tok2id.get("<PAD>", 0)
    unk_id = tok2id.get("<UNK>", 1)

    # invert
    id2tok = {}
    for t, i in tok2id.items():
        try:
            id2tok[int(i)] = t
        except Exception:
            pass

    return tok2id, id2tok, int(pad_id), int(unk_id)

def load_graph_id2token(id2tok_path=GRAPH_ID2TOKEN_JSON, fallback_tok2id_path=GRAPH_VOCAB_JSON):
    """
    Return: dict[int->str]
    """
    if os.path.isfile(id2tok_path):
        d = json.load(open(id2tok_path, "r", encoding="utf-8"))
        return {int(k): v for k, v in d.items()}

    # fallback: graph_vocab.json may contain {"token2id": {...}} or directly token2id
    tok2id = json.load(open(fallback_tok2id_path, "r", encoding="utf-8"))
    tok2id = tok2id.get("token2id", tok2id)
    return {int(v): k for k, v in tok2id.items()}

# ======================================================
# Dataset
# ======================================================

class MultiModalDatasetPT(Dataset):
    """
    Read graph .pt + seq_ids .pt
    - handle empty graph by dummy node
    - attach has_graph flag
    """
    def __init__(self, benign_graph_dir, ransomware_graph_dirs, seq_root, max_seq_len=1500):
        self.max_len = max_seq_len
        self.samples = []

        mapping = [(benign_graph_dir, os.path.join(seq_root, "benign"), 0)]
        for r_dir in ransomware_graph_dirs:
            mapping.append((r_dir, os.path.join(seq_root, "ransomware"), 1))

        for gdir, sdir, label in mapping:
            if not (os.path.isdir(gdir) and os.path.isdir(sdir)):
                print(f"[WARN] Missing dir: {gdir} or {sdir}")
                continue

            graph_map = {}
            for f in sorted(os.listdir(gdir)):
                if f.endswith(".pt"):
                    norm = normalize_name(f)
                    graph_map[norm] = os.path.join(gdir, f)

            for fname in sorted(os.listdir(sdir)):
                if not fname.endswith("_seq.pt"):
                    continue
                norm = normalize_name(fname)
                spath = os.path.join(sdir, fname)

                if norm in graph_map:
                    ppath = graph_map[norm]
                    self.samples.append((ppath, spath, label))

        self.samples.sort(key=lambda x: x[0])

        if not self.samples:
            raise RuntimeError("No (graph, seq) pairs found. Check filenames.")

        print(f"[INFO] Total samples: {len(self.samples)}")

        # infer feat_dim
        self.feat_dim = None
        for g_path, s_path, lbl in self.samples:
            data = torch.load(g_path, weights_only=False)
            if hasattr(data, "x") and data.x is not None and data.x.dim() == 2 and data.x.size(0) > 0:
                self.feat_dim = data.x.size(1)
                break
        if self.feat_dim is None:
            raise RuntimeError("Cannot infer feat_dim from graphs (all empty?).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ppath, spath, label = self.samples[i]
        data = torch.load(ppath, weights_only=False)
        seq_ids = torch.load(spath)  # 1D tensor [L] (or shorter)

        # empty graph -> dummy
        has_real_graph = True
        if (not hasattr(data, "x")) or (data.x is None) or (data.x.dim() != 2) or (data.x.size(0) == 0):
            has_real_graph = False
            data.x = torch.zeros((1, self.feat_dim), dtype=torch.float32)
            data.edge_index = torch.empty((2, 0), dtype=torch.long)

        data.has_graph = torch.tensor(1 if has_real_graph else 0, dtype=torch.long)

        # pad/trunc seq
        if self.max_len is not None:
            if seq_ids.size(0) > self.max_len:
                seq_ids = seq_ids[:self.max_len]
            else:
                pad_len = self.max_len - seq_ids.size(0)
                pad = torch.zeros(pad_len, dtype=seq_ids.dtype)
                seq_ids = torch.cat([seq_ids, pad], dim=0)

        return data, seq_ids.long(), torch.tensor(int(label), dtype=torch.float32)

def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(seqs), torch.stack(labels)

# ======================================================
# Model
# ======================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2   = BatchNorm(hidden)
        self.drop  = drop
        self.output_dim = hidden

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, self.drop, training=self.training)
        return global_mean_pool(x, batch)

class xLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed=128, seq_len=1500, blocks=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu"
                )
            ),
            context_length=seq_len,
            num_blocks=blocks,
            embedding_dim=embed,
            slstm_at=[0]
        )
        self.xlstm = xLSTMBlockStack(cfg)
        self.output_dim = embed
        self.seq_len = seq_len

    def forward(self, seq):
        out = self.xlstm(self.embed(seq))  # [B, L, D]
        return out.mean(dim=1)             # [B, D]

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hiddens=[128, 64], drop=0.3):
        super().__init__()
        layers = []
        dims = [in_dim] + hiddens
        for i in range(len(hiddens)):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(drop)
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)   # [B,1]
        return out.view(-1) # [B]

class MultiModalClassifier(nn.Module):
    """
    Same as training:
      - encoders
      - gate -> softmax -> w_seq/w_graph
      - fused = [w_seq*h_s, w_graph*h_g]
    """
    def __init__(self, graph_enc, seq_enc, fusion_h=128):
        super().__init__()
        self.graph_enc = graph_enc
        self.seq_enc   = seq_enc

        d_g = graph_enc.output_dim
        d_s = seq_enc.output_dim

        self.gate = nn.Linear(d_s + d_g, 2)
        self.fusion_dim = d_s + d_g
        self.classifier = MLPClassifier(self.fusion_dim, [fusion_h, fusion_h // 2])

    def forward(self, g, seq, return_branch=False, return_features=False):
        h_g = self.graph_enc(g.x, g.edge_index, g.batch)
        h_s = self.seq_enc(seq)

        if h_g.size(0) != h_s.size(0):
            b = min(h_g.size(0), h_s.size(0))
            h_g = h_g[:b]
            h_s = h_s[:b]
            if hasattr(g, "has_graph"):
                g.has_graph = g.has_graph[:b]

        B = h_s.size(0)
        if hasattr(g, "has_graph"):
            has_graph = g.has_graph.view(-1).to(h_g.device).float()
        else:
            has_graph = torch.ones(B, device=h_g.device)

        h_g = h_g * has_graph.view(-1, 1)

        gate_in = torch.cat([h_s, h_g], dim=-1)  # [B, d_s+d_g]
        scores = self.gate(gate_in)              # [B, 2]

        # if has_graph=0, force graph logit very small
        scores[:, 1] = scores[:, 1] + (has_graph - 1.0) * 1e4

        w = torch.softmax(scores, dim=-1)
        w_seq   = w[:, 0:1]
        w_graph = w[:, 1:2]

        h_fused = torch.cat([w_seq * h_s, w_graph * h_g], dim=-1)
        logits = self.classifier(h_fused)

        if (not return_branch) and (not return_features):
            return logits

        branch_info = {
            "w_seq": w_seq.detach().cpu(),
            "w_graph": w_graph.detach().cpu(),
            "has_graph": has_graph.detach().cpu()
        }

        if return_branch and (not return_features):
            return logits, branch_info
        if return_features and (not return_branch):
            return logits, h_fused
        if return_branch and return_features:
            return logits, h_fused, branch_info

        return logits
        
import matplotlib.pyplot as plt

# ======================================================
# LIME: build tokens from seq_ids, wrapper uses model with has_graph=0 dummy graph
# ======================================================

def seq_ids_to_tokens(seq_ids_1d, id2tok, pad_id=0, max_tokens_for_text=500):
    """
    Convert seq_ids -> list[str] tokens (ignore PAD).
    Limit for LIME text creation: max_tokens_for_text.
    """
    seq_ids_1d = seq_ids_1d.detach().cpu().numpy().tolist()
    toks = []
    for i in seq_ids_1d:
        if int(i) == int(pad_id):
            continue
        toks.append(id2tok.get(int(i), f"ID_{int(i)}"))
        if len(toks) >= max_tokens_for_text:
            break
    if not toks:
        toks = ["<EMPTY_SEQ>"]
    return toks

def encode_tokens_to_ids(tokens, tok2id, pad_id=0, unk_id=1, max_len=1500):
    ids = [tok2id.get(t, unk_id) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    return ids

class SeqPredictWrapperGated:
    """
    classifier_fn for LIME:
      - create dummy graph with has_graph=0 so model uses seq branch only.
      - input: list[str] (space-separated tokens) from LIME
      - output: np.array [N,2] probs for [class0, class1]
    """
    def __init__(self, model, feat_dim, tok2id, pad_id, unk_id, max_len, device, batch_size=16):
        self.m = model
        self.feat_dim = int(feat_dim)
        self.tok2id = tok2id
        self.pad_id = int(pad_id)
        self.unk_id = int(unk_id)
        self.max_len = int(max_len)
        self.device = device
        self.batch_size = int(batch_size)

    def _make_dummy_graph_batch(self, B: int):
        # 1 node per sample, no edges, has_graph=0
        data_list = []
        for _ in range(B):
            d = Data(
                x=torch.zeros((1, self.feat_dim), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )
            d.has_graph = torch.tensor(0, dtype=torch.long)
            data_list.append(d)
        return Batch.from_data_list(data_list).to(self.device)

    def __call__(self, texts):
        # texts: list[str]
        # tokenize by whitespace (tokens already are space separated)
        tok_lists = [t.strip().split() if isinstance(t, str) else list(t) for t in texts]

        out = []
        for i in range(0, len(tok_lists), self.batch_size):
            chunk = tok_lists[i:i+self.batch_size]
            ids = torch.tensor(
                [encode_tokens_to_ids(toks, self.tok2id, self.pad_id, self.unk_id, self.max_len) for toks in chunk],
                dtype=torch.long,
                device=self.device
            )
            g = self._make_dummy_graph_batch(B=ids.size(0))

            with torch.no_grad():
                logits = self.m(g, ids)  # [B]
                p1 = torch.sigmoid(logits).unsqueeze(1)  # [B,1]
                proba = torch.cat([1 - p1, p1], dim=1)   # [B,2]
            out.append(proba.detach().cpu().numpy())

            del ids, g, logits, p1, proba
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

        return np.concatenate(out, axis=0)

def topk_push_to_class(lime_list, k=5):
    """
    Với LIME as_list(label=c):
    - weight dương: đẩy về lớp c
    - weight âm: kéo ra khỏi lớp c
    lấy top-k weight dương (nếu thiếu thì fallback lấy top theo weight lớn nhất).
    """
    pos = [(t, float(w)) for t, w in lime_list if float(w) > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    if len(pos) >= k:
        return pos[:k]

    # fallback: lấy theo weight giảm dần để luôn có k item
    all_sorted = sorted([(t, float(w)) for t, w in lime_list], key=lambda x: x[1], reverse=True)
    return all_sorted[:k]
from matplotlib.patches import Patch

def plot_lime_top5_two_sides(top5_benign, top5_ransom, p_benign, p_ransom, k=5, save_path=None, sample_name=None):
    """
    Xen kẽ benign/ransom, hiển thị token thuần, giá trị lớn ở trên.
    - Benign: xanh dương, vẽ sang trái (âm)
    - Ransomware: đỏ, vẽ sang phải (dương)
    """
    top_b = (top5_benign or [])[:k]
    top_r = (top5_ransom or [])[:k]

    labels, vals, colors = [], [], []

    for i in range(k):
        # Benign i
        if i < len(top_b):
            tok, w = top_b[i]
            labels.append(str(tok))
            vals.append(-abs(float(w)))
        else:
            labels.append("(none)")
            vals.append(0.0)
        colors.append("royalblue")

        # Ransomware i
        if i < len(top_r):
            tok, w = top_r[i]
            labels.append(str(tok))
            vals.append(abs(float(w)))
        else:
            labels.append("(none)")
            vals.append(0.0)
        colors.append("red")

    plt.figure(figsize=(14, 7))
    y = np.arange(len(labels))
    plt.barh(y, vals, color=colors)
    plt.axvline(0, linewidth=1)

    plt.yticks(y, labels)
    plt.gca().invert_yaxis() 

    plt.xlabel("LIME weight")
    plt.title(f"Sample: {sample_name} | P(Benign)={p_benign:.4f}  P(Ransomware)={p_ransom:.4f}")

    # legend theo màu
    handles = [
        Patch(color="royalblue", label="Benign"),
        Patch(color="red", label="Ransomware"),
    ]
    plt.legend(handles=handles, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def explain_seq_lime_both(model, ds_item_seq_ids, id2tok_seq, tok2id_seq, feat_dim, max_len,
                          pad_id, unk_id, device,
                          top_k=20, num_samples=600, batch_size=16, max_tokens_for_text=500):
    instance_tokens = seq_ids_to_tokens(ds_item_seq_ids, id2tok_seq, pad_id=pad_id,
                                        max_tokens_for_text=max_tokens_for_text)
    text = " ".join(instance_tokens)

    wrapper = SeqPredictWrapperGated(
        model=model,
        feat_dim=feat_dim,
        tok2id=tok2id_seq,
        pad_id=pad_id,
        unk_id=unk_id,
        max_len=max_len,
        device=device,
        batch_size=batch_size
    )

    explainer = LimeTextExplainer(split_expression=lambda s: s.split(), bow=True)

    # Giải thích cho cả 2 lớp
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda texts: wrapper(texts),
        labels=[0, 1],
        num_features=top_k,
        num_samples=num_samples
    )

    benign_list = exp.as_list(label=0)      # token đẩy về benign
    ransom_list = exp.as_list(label=1)      # token đẩy về ransomware
    return benign_list, ransom_list, instance_tokens


# ======================================================
# GNNExplainer for graph branch (sequence fixed to zeros)
# ======================================================

def explain_graph_gnn(model, data_single, seq_len=1500, epochs=120, top_ratio=0.10, device=DEVICE):
    """
    Use wrapper that:
      - computes h_g from graph
      - sets h_s = zeros (no seq influence)
      - uses SAME gate + classifier so graph also influences w_graph via gate
    Return edge_mask, top_edges, node_score, top_nodes.
    """

    class GraphOnlyWrapper(nn.Module):
        def __init__(self, base_model, seq_len):
            super().__init__()
            self.m = base_model
            self.seq_len = int(seq_len)

        def forward(self, x, edge_index, batch):
            # B inferred from batch
            if batch is None or batch.numel() == 0:
                B = 1
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                B = int(batch.max().item()) + 1

            # graph encoder
            h_g = self.m.graph_enc(x, edge_index, batch)  # [B, d_g]

            # seq = zeros (PAD)
            h_s = torch.zeros((B, self.m.seq_enc.output_dim), device=x.device)

            has_graph = torch.ones(B, device=x.device)

            # gating as in model (scores[:,1] not penalized since has_graph=1)
            gate_in = torch.cat([h_s, h_g], dim=-1)     # [B, d_s+d_g]
            scores = self.m.gate(gate_in)               # [B,2]
            w = torch.softmax(scores, dim=-1)
            w_seq = w[:, 0:1]
            w_graph = w[:, 1:2]

            h_fused = torch.cat([w_seq * h_s, w_graph * h_g], dim=-1)
            logit = self.m.classifier(h_fused).unsqueeze(1)  # [B,1]

            # Explainer expects [B,2] for binary classification
            zeros = torch.zeros_like(logit)
            return torch.cat([zeros, logit], dim=1)          # [B,2]

    data = data_single.to(device)
    if (not hasattr(data, "batch")) or (data.batch is None):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    wrapper = GraphOnlyWrapper(model, seq_len=seq_len).to(device)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
    )

    explanation = explainer(x=data.x, edge_index=data.edge_index, batch=data.batch)
    edge_mask = explanation.edge_mask

    if edge_mask is None:
        return {
            "edge_mask": np.array([]),
            "top_edge_indices": [],
            "top_edges": [],
            "top_nodes": [],
            "node_score": [0.0] * int(data.num_nodes),
        }

    edge_mask = edge_mask.detach().cpu().numpy()
    m = len(edge_mask)
    k = max(1, int(m * top_ratio))
    top_idx = np.argsort(-edge_mask)[:k]

    top_edges = data.edge_index[:, top_idx].detach().cpu().numpy().T.tolist()

    node_score = np.zeros(int(data.num_nodes), dtype=float)
    ei = data.edge_index.t().detach().cpu().numpy()
    for (u, v), w in zip(ei, edge_mask):
        node_score[int(u)] += float(w)
        node_score[int(v)] += float(w)

    top_nodes = np.argsort(-node_score)[:max(1, int(data.num_nodes * top_ratio))].tolist()

    return {
        "edge_mask": edge_mask,
        "top_edge_indices": top_idx.tolist(),
        "top_edges": top_edges,
        "top_nodes": top_nodes,
        "node_score": node_score.tolist(),
    }

# ======================================================
# Main: predict + explain one sample
# ======================================================
import networkx as nx
import matplotlib.pyplot as plt

def _shorten(s: str, max_len: int = 70) -> str:
    s = str(s)
    return s if len(s) <= max_len else (s[:max_len-3] + "...")

def plot_gnnexplainer_subgraph(
    data_i,
    gnn_out: dict,
    idx2name_fn,                 # function: node_idx(int) -> str
    max_edges: int = 30,         # giới hạn số cạnh để đỡ rối
    max_nodes: int = 40,         # giới hạn số node để đỡ rối
    use_directed: bool = True,   # edge_index có hướng hay không
    with_labels: bool = True,
    draw_edge_labels: bool = False,
    save_path: str = None,
    title: str = None,
    seed: int = 42
):
    """
    Vẽ subgraph dựa trên top edges (theo edge_mask) từ gnn_out.

    YÊU CẦU:
    - data_i.edge_index: [2, E]
    - gnn_out["edge_mask"]: np.array[E]
    - gnn_out["top_edge_indices"]: list[eidx] (chỉ số cạnh theo thứ tự cột của edge_index)
    """

    top_edge_indices = gnn_out.get("top_edge_indices", []) or []
    edge_mask = gnn_out.get("edge_mask", None)

    if edge_mask is None or len(top_edge_indices) == 0:
        print("[plot_gnnexplainer_subgraph] Skip: no edge_mask/top edges.")
        return

    # Sort edges by importance desc, take max_edges
    pairs = [(int(eidx), float(edge_mask[int(eidx)])) for eidx in top_edge_indices]
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:max_edges]

    # Build subgraph edge list
    ei = data_i.edge_index.detach().cpu().numpy()  # [2,E]
    # Node set from selected edges
    node_set = set()
    edge_list = []
    for eidx, w in pairs:
        u = int(ei[0, eidx])
        v = int(ei[1, eidx])
        node_set.add(u); node_set.add(v)
        edge_list.append((u, v, w))

    # If too many nodes, keep the most "connected" ones (by sum edge weights)
    if len(node_set) > max_nodes:
        score = {n: 0.0 for n in node_set}
        for u, v, w in edge_list:
            score[u] += w
            score[v] += w
        keep = sorted(score.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep_nodes = set([n for n, _ in keep])
        edge_list = [(u, v, w) for (u, v, w) in edge_list if (u in keep_nodes and v in keep_nodes)]
        node_set = keep_nodes

    G = nx.DiGraph() if use_directed else nx.Graph()

    # Add nodes with labels
    for n in node_set:
        G.add_node(n, label=_shorten(idx2name_fn(int(n))))

    # Add edges with weights
    for u, v, w in edge_list:
        G.add_edge(u, v, weight=float(w))

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Edge widths scaled by weight
    ws = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(ws) if ws else 1.0
    widths = [1.0 + 4.0 * (w / max_w) for w in ws]  # 1..5

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=900)  # màu mặc định
    nx.draw_networkx_edges(G, pos, width=widths, arrows=use_directed, arrowsize=16)

    if with_labels:
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    if draw_edge_labels:
        e_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=7)

    if title is None:
        title = f"GNNExplainer subgraph (top_edges={len(edge_list)})"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
import networkx as nx
import matplotlib.pyplot as plt

def draw_gnnexplainer_subgraph_with_table(
    data_i,
    gnn_out: dict,
    idx2name_fn,                  # function: node_idx(int) -> str
    max_edges: int = 30,
    max_nodes: int = 40,
    max_table_rows: int = 40,     # số dòng trong bảng chú thích
    use_directed: bool = True,
    draw_edge_labels: bool = False,
    edge_label_mode: str = "weight",  # "weight" hoặc "rank"
    save_path: str = None,
    title: str = None,
    seed: int = 42,
    name_max_len: int = 90,
    wrap_width: int = 70
):
    """
    Vẽ subgraph (bên trái) + bảng chú thích node_id -> node_name (bên phải) trong 1 figure.
    - Node trên graph hiển thị bằng node_id (chính là node_idx).
    - Bảng: sắp theo node_score (tổng edge_mask quanh node) giảm dần.
    """

    top_edge_indices = gnn_out.get("top_edge_indices", []) or []
    edge_mask = gnn_out.get("edge_mask", None)

    if edge_mask is None or len(edge_mask) == 0:
        print("[draw_gnnexplainer_subgraph_with_table] Skip: empty edge_mask.")
        return

    # Nếu top_edge_indices rỗng (hiếm), fallback dùng toàn bộ cạnh
    if len(top_edge_indices) == 0:
        top_edge_indices = list(range(len(edge_mask)))

    # Sort edges by importance desc, take max_edges
    pairs = [(int(eidx), float(edge_mask[int(eidx)])) for eidx in top_edge_indices]
    pairs.sort(key=lambda x: x[1], reverse=True)
    if max_edges is not None:
        pairs = pairs[:max_edges]

    ei = data_i.edge_index.detach().cpu().numpy()  # [2,E]

    # Build edge_list + node_set
    node_set = set()
    edge_list = []
    for eidx, w in pairs:
        u = int(ei[0, eidx]); v = int(ei[1, eidx])
        node_set.add(u); node_set.add(v)
        edge_list.append((u, v, float(w), int(eidx)))

    if len(edge_list) == 0:
        print("[draw_gnnexplainer_subgraph_with_table] Skip: no edges after filtering.")
        return

    # Compute node_score from selected edges
    node_score = {n: 0.0 for n in node_set}
    for u, v, w, _ in edge_list:
        node_score[u] += w
        node_score[v] += w

    # Limit nodes if too many: keep top by node_score
    if max_nodes is not None and len(node_set) > max_nodes:
        keep_nodes = set([n for n, _ in sorted(node_score.items(), key=lambda x: x[1], reverse=True)[:max_nodes]])
        edge_list = [(u, v, w, eidx) for (u, v, w, eidx) in edge_list if (u in keep_nodes and v in keep_nodes)]
        node_set = keep_nodes
        node_score = {n: node_score[n] for n in node_set}

    # Build networkx graph
    G = nx.DiGraph() if use_directed else nx.Graph()
    for n in node_set:
        G.add_node(n)

    for u, v, w, eidx in edge_list:
        G.add_edge(u, v, weight=w, eidx=eidx)

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Edge widths scaled by weight
    ws = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(ws) if ws else 0.0
    if max_w <= 1e-12:
        widths = [1.5 for _ in ws]          # constant width
    else:
        widths = [1.0 + 4.0 * (w / max_w) for w in ws]  # 1..5

    # ======== Build 1 figure with 2 panels ========
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.2])
    ax_g = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_t.axis("off")

    # Draw graph (LEFT)
    nx.draw_networkx_nodes(G, pos, node_size=900, ax=ax_g)
    nx.draw_networkx_edges(G, pos, width=widths, arrows=use_directed, arrowsize=16, ax=ax_g)

    # Node labels = node id
    node_labels = {n: str(int(n)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax_g)

    # Optional edge labels
    if draw_edge_labels:
        if edge_label_mode == "rank":
            # rank theo thứ tự importance (dễ nhìn hơn weight gần giống nhau)
            rank_map = {}
            # sort edges by weight desc
            sorted_edges = sorted([(u, v, G[u][v]["weight"]) for u, v in G.edges()],
                                  key=lambda x: x[2], reverse=True)
            for r, (u, v, _) in enumerate(sorted_edges, start=1):
                rank_map[(u, v)] = str(r)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=rank_map, font_size=7, ax=ax_g)
        else:
            e_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=7, ax=ax_g)

    if title is None:
        title = f"GNNExplainer subgraph (nodes={len(G.nodes())}, edges={len(G.edges())})"
    ax_g.set_title(title)
    ax_g.axis("off")

    # ======== Table (RIGHT): node_id -> meaning ========
    rows = []
    # sort nodes by node_score desc
    sorted_nodes = sorted(node_score.items(), key=lambda x: x[1], reverse=True)

    for n, sc in sorted_nodes[:max_table_rows]:
        raw_name = idx2name_fn(int(n))

        # nếu cắt bớt, còn không thì set name_max_len=None khi gọi
        if name_max_len is not None:
            raw_name = _shorten(raw_name, max_len=name_max_len)

        # wrap text theo nhiều dòng để hiện hết nội dung
        name = textwrap.fill(raw_name, width=wrap_width)

        rows.append([str(int(n)), name, f"{sc:.3f}"])

    col_labels = ["node_id", "node_meaning", "score"]
    # node_id hẹp, node_meaning rộng, score hẹp
    col_widths = [0.12, 0.76, 0.12]

    table = ax_t.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def robust_load_state_dict(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        # common patterns
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                model.load_state_dict(obj[k])
                return obj
        # or direct state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            model.load_state_dict(obj)
            return obj
    # fallback
    model.load_state_dict(obj)
    return obj

def predict_and_explain_one(
    model, ds, idx, device,
    tok2id_seq, id2tok_seq, pad_id, unk_id,
    graph_id2tok,
    save_prefix=SAVE_PREFIX,
    seq_len=1500,
    lime_top_k=20, lime_num_samples=600, lime_batch_size=16, lime_max_tokens_for_text=500,
    gnn_epochs=120, gnn_top_ratio=0.10
):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    # --- show file paths being analyzed
    g_path, s_path, _ = ds.samples[idx]
    print(f"[FILE] graph_pt = {g_path}")
    print(f"[FILE] seq_pt   = {s_path}")
    sample_name = normalize_name(os.path.basename(g_path))
    # --- load sample
    data, seq_ids, label = ds[idx]
    label = int(label.item())

    g = Batch.from_data_list([data]).to(device)
    seq = seq_ids.unsqueeze(0).to(device)  # [1,L]

    model.eval()
    with torch.no_grad():
        # --- FULL
        logits, br = model(g, seq, return_branch=True)
        prob_full = torch.sigmoid(logits)[0].item()
        pred_full = int(prob_full >= 0.5)
    
        w_seq   = br["w_seq"][0].item()
        w_graph = br["w_graph"][0].item()
        has_graph = int(br["has_graph"][0].item())
    
        # ======================================================
        # SEQ-ONLY (clone g để không làm bẩn g)
        # ======================================================
        g_seqonly = g.clone()  # <<< quan trọng
        if hasattr(g_seqonly, "has_graph"):
            g_seqonly.has_graph = torch.zeros_like(g_seqonly.has_graph)
        else:
            g_seqonly.has_graph = torch.zeros(1, dtype=torch.long, device=device)
    
        logits_seqonly, br_seqonly = model(g_seqonly, seq, return_branch=True)
        prob_seqonly = torch.sigmoid(logits_seqonly)[0].item()
    
        # ======================================================
        # GRAPH-ONLY (dùng g "sạch", không bị set has_graph=0)
        # ======================================================
        seq_zeros = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    
        g_graphonly = g.clone()  # <<< cũng nên clone cho chắc
        if hasattr(g_graphonly, "has_graph"):
            g_graphonly.has_graph = torch.ones_like(g_graphonly.has_graph)  # đảm bảo graph bật
        else:
            g_graphonly.has_graph = torch.ones(1, dtype=torch.long, device=device)
    
        logits_graphonly, br_graphonly = model(g_graphonly, seq_zeros, return_branch=True)
        prob_graphonly = torch.sigmoid(logits_graphonly)[0].item()


    print(f"[Inference] idx={idx} label={label} | prob_full={prob_full:.4f} pred={pred_full}")
    print(f"[Gate(full)] has_graph={has_graph} | w_seq={w_seq:.3f} w_graph={w_graph:.3f}")
    print(f"[Ablation] prob_seq_only={prob_seqonly:.4f} | prob_graph_only={prob_graphonly:.4f}")
    print(f"[GraphOnly check] has_graph={int(br_graphonly['has_graph'][0].item())} "
      f"w_seq={br_graphonly['w_seq'][0].item():.3f} w_graph={br_graphonly['w_graph'][0].item():.3f}")

    
    # ======================================================
    # (1) LIME for sequence (both classes)
    # ======================================================
    lime_benign, lime_ransom, instance_tokens = explain_seq_lime_both(
        model=model,
        ds_item_seq_ids=seq_ids,
        id2tok_seq=id2tok_seq,
        tok2id_seq=tok2id_seq,
        feat_dim=ds.feat_dim,
        max_len=seq_len,
        pad_id=pad_id,
        unk_id=unk_id,
        device=device,
        top_k=lime_top_k,
        num_samples=lime_num_samples,
        batch_size=lime_batch_size,
        max_tokens_for_text=lime_max_tokens_for_text
    )

    # --- Save CSV
    with open(f"{save_prefix}_sequence_lime_benign.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token", "lime_weight_for_benign"])
        for r, (tok, wgt) in enumerate(lime_benign, 1):
            w.writerow([r, tok, float(wgt)])

    with open(f"{save_prefix}_sequence_lime_ransomware.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token", "lime_weight_for_ransomware"])
        for r, (tok, wgt) in enumerate(lime_ransom, 1):
            w.writerow([r, tok, float(wgt)])

    print("[XAI-Seq LIME benign] preview:", ", ".join([f"{t}:{w:+.3f}" for t, w in lime_benign[:10]]))
    print("[XAI-Seq LIME ransom] preview:", ", ".join([f"{t}:{w:+.3f}" for t, w in lime_ransom[:10]]))
    print(f"[SAVE] {save_prefix}_sequence_lime_benign.csv")
    print(f"[SAVE] {save_prefix}_sequence_lime_ransomware.csv")

    # --- Probabilities
    p_ransom = float(prob_full)
    p_benign = float(1.0 - prob_full)
    print(f"[PROB] P(Benign)={p_benign:.4f} | P(Ransomware)={p_ransom:.4f}")

    # --- Top-5 mỗi phía (đúng nghĩa “đẩy về benign / đẩy về ransomware”)
    top5_benign = topk_push_to_class(lime_benign, k=5)
    top5_ransom = topk_push_to_class(lime_ransom, k=5)

    print("\n[TOP-5] Feature đẩy về BENIGN (LIME label=0, weight dương):")
    for i, (t, wgt) in enumerate(top5_benign, 1):
        print(f"  {i:02d}. {t}  ({wgt:+.4f})")

    print("\n[TOP-5] Feature đẩy về RANSOMWARE (LIME label=1, weight dương):")
    for i, (t, wgt) in enumerate(top5_ransom, 1):
        print(f"  {i:02d}. {t}  ({wgt:+.4f})")

    # --- Plot 2 bên (Benign xanh, Ransom đỏ)
    plot_path = f"{save_prefix}_lime_top5_two_sides.png"
    plot_lime_top5_two_sides(top5_benign, top5_ransom, p_benign, p_ransom, k=5, save_path=plot_path, sample_name=sample_name)
    print(f"[SAVE] {plot_path}")

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


    # ======================================================
    # (2) GNNExplainer for graph
    # ======================================================
    data_i = g.to_data_list()[0]
    if (not hasattr(data_i, "batch")) or (data_i.batch is None):
        data_i.batch = torch.zeros(data_i.x.size(0), dtype=torch.long, device=device)

    # If graph is dummy/no edges -> explanation is not meaningful
    graph_ok = True
    if (not hasattr(data_i, "edge_index")) or (data_i.edge_index is None) or (data_i.edge_index.numel() == 0):
        graph_ok = False
    if has_graph == 0:
        graph_ok = False

    gnn_out = None
    top_nodes_named, top_edges_named = [], []

    if graph_ok:
        gnn_out = explain_graph_gnn(
            model=model,
            data_single=data_i,
            seq_len=seq_len,
            epochs=gnn_epochs,
            top_ratio=gnn_top_ratio,
            device=device
        )

        # map node idx -> node_ids -> token name
        def idx2name(node_idx: int) -> str:
            if hasattr(data_i, "node_ids"):
                nid = int(data_i.node_ids[int(node_idx)])
                return graph_id2tok.get(nid, f"UNK:{nid}")
            return f"node_{int(node_idx)}"

        top_nodes_named = [
            (int(nidx), idx2name(int(nidx)), float(gnn_out["node_score"][int(nidx)]))
            for nidx in gnn_out["top_nodes"]
        ]

        for r, (u, v) in enumerate(gnn_out["top_edges"], 1):
            eidx = gnn_out["top_edge_indices"][r-1]
            emask = float(gnn_out["edge_mask"][eidx]) if len(gnn_out["edge_mask"]) else 0.0
            top_edges_named.append({
                "rank": r,
                "src_idx": int(u), "dst_idx": int(v),
                "src_name": idx2name(int(u)), "dst_name": idx2name(int(v)),
                "edge_mask": emask
            })

        with open(f"{save_prefix}_graph_nodes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["node_idx", "node_name", "score"])
            for row in top_nodes_named:
                w.writerow(row)

        with open(f"{save_prefix}_graph_edges.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank","src_idx","dst_idx","src_name","dst_name","edge_mask"])
            w.writeheader()
            for row in top_edges_named:
                w.writerow(row)

        print("[XAI-Graph] nodes preview:", top_nodes_named[:5])
        print("[XAI-Graph] edges preview:", top_edges_named[:5])
        print(f"[SAVE] {save_prefix}_graph_nodes.csv")
        print(f"[SAVE] {save_prefix}_graph_edges.csv")

        # --- Vẽ subgraph từ GNNExplainer
        subgraph_path = f"{save_prefix}_gnn_subgraph.png"
        plot_gnnexplainer_subgraph(
            data_i=data_i,
            gnn_out=gnn_out,
            idx2name_fn=idx2name,
            max_edges=50,          # 20/30/50
            max_nodes=40,
            use_directed=True,
            with_labels=True,
            draw_edge_labels=False,
            save_path=subgraph_path,
            title=f"GNNExplainer Subgraph | sample={sample_name}"
        )
        print(f"[SAVE] {subgraph_path}")
        draw_gnnexplainer_subgraph_with_table(
            data_i=data_i,
            gnn_out=gnn_out,
            idx2name_fn=idx2name,
            max_edges=50,          # 20/30/50... hoặc 10**9 để lấy toàn bộ
            max_nodes=40,          # tương tự
            max_table_rows=40,     # số dòng bảng chú thích
            use_directed=True,
            draw_edge_labels=False,        # nếu muốn bật thì xem note bên dưới
            edge_label_mode="rank",        # "rank" thường dễ đọc hơn "weight"
            save_path=subgraph_path,
            title=f"GNNExplainer Subgraph | sample={sample_name}",
            name_max_len=None,     # <<< không cắt, giữ nguyên chuỗi
            wrap_width=70  
        )
    else:
        print("[XAI-Graph] SKIP (no real graph / no edges / has_graph=0)")

    # ======================================================
    # (3) Branch summary (use gating weights directly)
    # ======================================================
    dominant = "seq" if w_seq >= w_graph else "graph"

    with open(f"{save_prefix}_branch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index","label","prob_full","pred_full",
            "has_graph","w_seq","w_graph",
            "prob_seq_only","prob_graph_only",
            "dominant"
        ])
        w.writeheader()
        w.writerow({
            "index": idx,
            "label": label,
            "prob_full": prob_full,
            "pred_full": pred_full,
            "has_graph": has_graph,
            "w_seq": w_seq,
            "w_graph": w_graph,
            "prob_seq_only": prob_seqonly,
            "prob_graph_only": prob_graphonly,
            "dominant": dominant
        })
    print(f"[XAI-Branch] dominant={dominant} | w_seq={w_seq:.3f} w_graph={w_graph:.3f}")
    print(f"[SAVE] {save_prefix}_branch.csv")

    return {
        "index": idx,
        "label": label,
        "prob_full": prob_full,
        "pred_full": pred_full,
        "gate": {
            "has_graph": has_graph,
            "w_seq": w_seq,
            "w_graph": w_graph,
            "dominant": dominant
        },
        "ablation": {
            "prob_seq_only": prob_seqonly,
            "prob_graph_only": prob_graphonly
        },
        "seq_lime_benign": lime_benign,
        "seq_lime_ransom": lime_ransom,
        "seq_lime": lime_ransom,
        "graph_top_nodes_named": top_nodes_named,
        "graph_top_edges_named": top_edges_named,
        "file_info": {
            "graph_path": g_path,
            "seq_path": s_path
        },
        "sample_name": sample_name,
    }

# ======================================================
# LLM helpers: build payload + call OpenAI
# ======================================================

def _label_name(v: int) -> str:
    # 0 = benign, 1 = ransomware
    return "ransomware" if int(v) == 1 else "benign"

def build_llm_payload_from_out(out_dict):
    """
    Convert the raw output of predict_and_explain_one(...) into a JSON-friendly
    payload for the LLM.
    """
    label = int(out_dict["label"])
    pred  = int(out_dict["pred_full"])
    prob  = float(out_dict["prob_full"])

    gate = out_dict.get("gate", {})
    abl  = out_dict.get("ablation", {})

    seq_items = []
    for rank, (tok, wgt) in enumerate(out_dict.get("seq_lime", []), start=1):
        seq_items.append({
            "rank": rank,
            "token": str(tok),
            "weight": float(wgt),
        })

    graph_nodes = []
    for node_idx, node_name, score in out_dict.get("graph_top_nodes_named", []):
        graph_nodes.append({
            "node_idx": int(node_idx),
            "node_name": str(node_name),
            "score": float(score),
        })

    graph_edges = []
    for e in out_dict.get("graph_top_edges_named", []):
        graph_edges.append({
            "rank": int(e.get("rank", 0)),
            "src_idx": int(e.get("src_idx", -1)),
            "dst_idx": int(e.get("dst_idx", -1)),
            "src_name": str(e.get("src_name", "")),
            "dst_name": str(e.get("dst_name", "")),
            "edge_weight": float(e.get("edge_mask", 0.0)),
        })

    file_info = out_dict.get("file_info", {})

    payload = {
        "sample_index": int(out_dict.get("index", -1)),
        "files": {
            "graph_path": file_info.get("graph_path", ""),
            "seq_path": file_info.get("seq_path", "")
        },
        "true_label": label,
        "true_label_name": _label_name(label),
        "pred_label": pred,
        "pred_label_name": _label_name(pred),
        "predicted_probability_ransomware": prob,
        "model_fusion": {
            "sequence_strength": float(gate.get("w_seq", 0.0)),
            "graph_strength": float(gate.get("w_graph", 0.0)),
            "dominant_branch": str(gate.get("dominant", "unknown")),
            "prob_seq_only": float(abl.get("prob_seq_only", 0.0)),
            "prob_graph_only": float(abl.get("prob_graph_only", 0.0)),
        },
        "sequence_lime_tokens": seq_items,
        "graph_top_nodes": graph_nodes,
        "graph_top_edges": graph_edges,
        "token_semantics": {
            "sequence_token_prefixes": {
                "api:": "Windows API call",
                "feature:": "Aggregated behavior feature (registry, file, etc.)",
                "dropped_file:": "File dropped or created on disk",
                "signature:": "Sandbox heuristic signature",
                "process:": "Windows process name",
                "network:": "Network activity (protocol, destination, port)",
            },
            "graph_token_prefixes": {
                "api:": "API call node in the behavior graph",
                "feature:": "Feature node (registry, file, string, etc.)",
                "process:": "Process node (name, path, cmdline)",
                "dropped:": "Dropped-file node",
                "network:": "Network-activity node",
                "signature:": "Sandbox signature node",
            },
        },
    }
    return payload


def generate_llm_general_user_report(payload,
                                     model_name=None,
                                     temperature=0.2):
    """
    Generate a general-audience (non-technical) report from the XAI payload.
    Output is intended to be dashboard-friendly:
      - PART A: a single JSON object in a json fenced block
      - PART B: a short Markdown report in English
    """
    if not _HAS_OPENAI:
        raise RuntimeError("openai client (from 'openai' package) not installed; cannot call LLM.")

    client = OpenAI()

    if model_name is None:
        model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4.1-mini")

    system_prompt = textwrap.dedent("""        You are a security assistant explaining a ransomware detection result to a NON-TECHNICAL user.

        Goal:
        - Help them understand the risk level and what to do next.
        - Use plain language. Avoid jargon, API names, registry paths, or command lines.
        - Be calm and practical. Do NOT scare the user.
        - NEVER provide instructions for creating malware, evading security tools, or hacking.

        You will receive a JSON input describing:
        - model prediction (label), ransomware probability, confidence
        - optional branch weights (w_seq, w_graph)
        - summarized XAI signals (LIME, GNNExplainer) and a few high-level hints

        IMPORTANT:
        - Do not dump raw input.
        - No URLs.
        - Keep bullets short.

        OUTPUT FORMAT (STRICT):
        PART A) Return EXACTLY ONE JSON object inside a fenced block: ```json ... ```
        Use this schema:
        {
          "risk_level": "low|medium|high|critical",
          "headline": "one short sentence",
          "confidence": 0.0,
          "what_it_means": "1-2 short sentences",
          "top_signals": ["3-6 bullets in everyday language"],
          "next_steps": ["3-6 safe immediate steps a normal user can do"],
          "when_to_seek_help": ["2-4 cases to contact IT/support"],
          "limitations": "1-2 sentences about uncertainty/false positives"
        }

        Risk mapping based on ransomware_probability:
        - < 0.40 => low
        - 0.40–0.70 => medium
        - 0.70–0.90 => high
        - >= 0.90 => critical

        Rules for top_signals:
        - Describe behaviors in human terms (e.g., “tries to remove backups”, “many file changes quickly”).
        - Do NOT include raw API names or exact command lines.
        - Prefer <= 12 words per bullet.

        Rules for next_steps:
        - Only safe defensive steps: disconnect from network, do not run the file, scan with trusted AV,
          back up important files, contact IT, etc.
        - No offensive or evasion content.

        PART B) A short Markdown report (<= 250 words) IN ENGLISH with these headings:
        ### Result
        ### Why the system thinks this
        ### What you should do now
        ### Technical details (optional)

        Do not repeat the full JSON verbatim.
    """)

    user_prompt = "Here is the input JSON:\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content

def generate_llm_markdown_explanation(payload,
                                      model_name=None,
                                      temperature=0.2):
    """
    Call an OpenAI-compatible chat model to turn the XAI payload into a
    Markdown explanation. Controlled by ENABLE_LLM_XAI env var.
    """
    if not _HAS_OPENAI:
        raise RuntimeError("openai client (from 'openai' package) not installed; cannot call LLM.")

    client = OpenAI()

    if model_name is None:
        model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4.1-mini")

    system_prompt = textwrap.dedent("""
        You are a cybersecurity assistant that explains the decisions of a ransomware detection model
        to security analysts, developers, and advanced users.

        Model:
        - A multimodal detector for Windows samples.
        - It has two branches: an xLSTM sequence branch and a GCN behavior-graph branch.
        - The final decision comes from a fusion of both branches.

        Input format:
        You receive ONE JSON object containing:
        - sample_index, files.{graph_path, seq_path}
        - true_label, true_label_name
        - pred_label, pred_label_name
        - predicted_probability_ransomware
        - model_fusion: { sequence_strength, graph_strength, dominant_branch,
                         prob_seq_only, prob_graph_only }
        - sequence_lime_tokens: { rank, token, weight }
        - graph_top_nodes: { node_idx, node_name, score }
        - graph_top_edges: { rank, src_name, dst_name, edge_weight }
        - token_semantics: description of token prefixes

        Token syntax:
        - SEQUENCE branch:
          * "api:NAME" → Windows API call NAME.
          * "feature:TYPE:VALUE" → behavior feature (e.g., registry path, file path).
          * "dropped_file:PATH" → file dropped/created on disk.
          * "signature:NAME" → sandbox heuristic signature.
          * "process:NAME" → process name.
          * "network:PROTO:DST:PORT" → network connection summary.

        - GRAPH branch:
          * "api:NAME" → API call node.
          * "feature:TYPE:VALUE" → feature node.
          * "process:NAME:PATH:CMD" → process node.
          * "dropped:PATH" → dropped-file node.
          * "network:CATEGORY" → network node.
          * "signature:NAME" → signature node.

        Your goals:
        1. Explain WHY the model predicted "ransomware" or "benign" for this sample.
        2. Clearly state whether the prediction matches the ground truth.
        3. Comment on which branch (sequence vs graph) dominated the decision, using
           both the gating weights and the branch-only probabilities.
        4. Turn LIME and GNNExplainer outputs into a security-focused, human-readable
           explanation:
           - Group important sequence tokens into behaviors: APIs, registry, files,
             processes, network, signatures.
           - Describe the most important graph nodes and edges as relationships
             (e.g., process X accessing registry Y or dropping file Z).
        5. Use clear language that malware analysts can trust, but that developers or
           advanced users can follow.
        6. FP/FN risk estimation rules of thumb:
            - If pred = ransomware:
                FP risk increases when:
                - prob_full is not high or close to 0.5
                - LIME/GNN evidence supporting ransomware is weak
                - strong evidence pushes toward benign (LIME benign side is strong)
                - both prob_seq_only and prob_graph_only do not support ransomware
                - graph is missing/empty (has_graph=0) or very low node/edge count
            - If pred = benign:
                FN risk increases when:
                - LIME shows strong ransomware-pushing tokens (signatures/crypto/file ops/network/process hints)
                - GNNExplainer highlights suspicious behavior-related nodes/edges
                - one branch (seq-only or graph-only) is confident toward ransomware but fusion outputs benign
                - signals are contradictory (mixed dominance)

        Output format:
        - Always respond in Markdown.
        - Use this section outline:

          ## 1. High-level verdict
          ## 2. Branch contributions (fusion)
          ## 3. Sequence branch (LIME)
          ## 4. Graph branch (GNNExplainer)
          ## 5. Overall security assessment

        Do NOT dump the raw JSON back to the user. Summarise it and explain it.
    """)

    user_prompt = textwrap.dedent(f"""
        Here is the model prediction and XAI explanation for one Windows sample,
        in the JSON format described in the system message:

        ```json
        {json.dumps(payload, indent=2)}
        ```

        Please generate the Markdown explanation now.
    """)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # ======================================================
    # RUN
    # ======================================================

    # 1) Load dataset
    ds = MultiModalDatasetPT(
        BENIGN_GRAPH_DIR,
        RANSOMWARE_GRAPH_DIRS,
        SEQ_ROOT,
        max_seq_len=1500
    )

    # 2) Load vocabs
    tok2id_seq, id2tok_seq, PAD_ID, UNK_ID = load_seq_vocab(SEQ_VOCAB_JSON)
    graph_id2tok = load_graph_id2token(GRAPH_ID2TOKEN_JSON, GRAPH_VOCAB_JSON)

    print(f"[INFO] Seq vocab size={len(tok2id_seq)} | PAD={PAD_ID} UNK={UNK_ID}")
    print(f"[INFO] Graph id2token size={len(graph_id2tok)}")

    # 3) Build model and load weights
    # infer in_feats from dataset
    sample_g, sample_seq, _ = ds[0]
    in_feats = sample_g.x.size(1)
    seq_len = 1500
    vocab_size = len(tok2id_seq)

    graph_enc = GCNEncoder(in_feats).to(DEVICE)
    seq_enc   = xLSTMEncoder(vocab_size=vocab_size, seq_len=seq_len).to(DEVICE)
    model     = MultiModalClassifier(graph_enc, seq_enc).to(DEVICE)

    _ = robust_load_state_dict(model, MODEL_PT, DEVICE)
    model.eval()
    print(f"[INFO] Loaded model: {MODEL_PT}")

    # 4) Pick index safely
    if INDEX >= len(ds):
        print(f"[WARN] INDEX={INDEX} out of range -> using 0")
        INDEX = 0

    # 5) Explain
    out = predict_and_explain_one(
        model=model,
        ds=ds,
        idx=INDEX,
        device=DEVICE,
        tok2id_seq=tok2id_seq,
        id2tok_seq=id2tok_seq,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        graph_id2tok=graph_id2tok,
        save_prefix=SAVE_PREFIX,
        seq_len=seq_len,
        lime_top_k=LIME_TOP_K,
        lime_num_samples=LIME_NUM_SAMPLES,
        lime_batch_size=LIME_BATCH_SIZE,
        lime_max_tokens_for_text=LIME_MAX_TOKENS_FOR_TEXT,
        gnn_epochs=GNN_EPOCHS,
        gnn_top_ratio=GNN_TOP_RATIO
    )

    print("\n[DONE] XAI output keys:", list(out.keys()))

    # 6) (Optional) LLM explanation
    ENABLE_LLM = os.environ.get("ENABLE_LLM_XAI", "0") == "1"

    if ENABLE_LLM:
        try:
            payload = build_llm_payload_from_out(out)
            md_report = generate_llm_markdown_explanation(payload)
            md_path = f"{SAVE_PREFIX}_llm_report.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_report)
            print(f"[LLM] Markdown explanation saved to: {md_path}")
        except Exception as e:
            print("[LLM] Failed to generate LLM explanation:", repr(e))
    else:
        print("[LLM] Skipped (set ENABLE_LLM_XAI=1 to enable LLM explanations).")