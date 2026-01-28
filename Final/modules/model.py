import os
import re
import csv
import json

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# xLSTM
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig
)

# ======================================================
# Dataset
# ======================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def normalize_name(name):
    """
    Chuẩn hoá tên file:
    - Bỏ đuôi _seq.pt hoặc .pt
    - Bỏ phần .exe, .EXE, .ex, .EX...
    """
    name = name.replace("_seq.pt", "")
    name = name.replace(".pt", "")
    name = re.sub(r"\.(e?xe?)$", "", name, flags=re.IGNORECASE)
    return name


class MultiModalDatasetPT(Dataset):
    """
    Dataset đọc graph .pt + seq_ids .pt
    - Tự xử lý graph rỗng bằng cách tạo dummy node
    - Gắn thêm cờ has_graph cho mỗi sample
    """
    def __init__(self, benign_graph_dir, ransomware_graph_dirs, seq_root, max_seq_len=1500):
        self.max_len = max_seq_len
        self.samples = []

        mapping = [
            (benign_graph_dir, os.path.join(seq_root, "benign"), 0),
        ]
        for r_dir in ransomware_graph_dirs:
            mapping.append((r_dir, os.path.join(seq_root, "ransomware"), 1))

        # Gom tất cả cặp (graph, seq, label)
        for gdir, sdir, label in mapping:
            if not (os.path.isdir(gdir) and os.path.isdir(sdir)):
                print(f"[WARN] Thiếu thư mục: {gdir} hoặc {sdir}")
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
            raise RuntimeError("Không tìm thấy cặp graph + seq nào. Kiểm tra tên file.")

        print(f"[INFO] Tổng số sample: {len(self.samples)}")

        # Suy ra feat_dim từ một graph hợp lệ
        self.feat_dim = None
        for g_path, s_path, lbl in self.samples:
            data = torch.load(g_path, weights_only=False)
            if hasattr(data, "x") and data.x is not None and data.x.dim() == 2 and data.x.size(0) > 0:
                self.feat_dim = data.x.size(1)
                break

        if self.feat_dim is None:
            raise RuntimeError("Không tìm được graph hợp lệ để suy ra feat_dim!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ppath, spath, label = self.samples[i]

        data = torch.load(ppath, weights_only=False)
        seq_ids = torch.load(spath)

        # Xử lý graph rỗng: tạo dummy node + cờ has_graph
        has_real_graph = True
        if (not hasattr(data, "x")) or (data.x is None) or (data.x.dim() != 2) or (data.x.size(0) == 0):
            has_real_graph = False
            data.x = torch.zeros((1, self.feat_dim), dtype=torch.float32)
            data.edge_index = torch.empty((2, 0), dtype=torch.long)

        data.has_graph = torch.tensor(1 if has_real_graph else 0, dtype=torch.long)

        # Pad/cắt sequence
        if self.max_len is not None:
            if seq_ids.size(0) > self.max_len:
                seq_ids = seq_ids[:self.max_len]
            else:
                pad_len = self.max_len - seq_ids.size(0)
                pad = torch.zeros(pad_len, dtype=seq_ids.dtype)
                seq_ids = torch.cat([seq_ids, pad], dim=0)

        return data, seq_ids, torch.tensor(int(label), dtype=torch.float32)


def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(seqs), torch.stack(labels)


# ======================================================
# Encoders & Fusion classifier với gating w_seq / w_graph
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
        out = self.mlp(x)       # [B, 1]
        return out.view(-1)     # [B] luôn, kể cả B=1


class MultiModalClassifier(nn.Module):
    """
    F: encoder chung cho cả source/target
       - GCNEncoder + xLSTMEncoder
       - Gating w_seq / w_graph
    """
    def __init__(self, graph_enc, seq_enc, fusion_h=128):
        super().__init__()
        self.graph_enc = graph_enc
        self.seq_enc   = seq_enc

        d_g = graph_enc.output_dim
        d_s = seq_enc.output_dim

        # Gate: cho ra 2 logit (seq, graph)
        self.gate = nn.Linear(d_s + d_g, 2)

        fusion_dim = d_g + d_s
        self.fusion_dim = fusion_dim   # dùng cho DomainDiscriminator
        self.classifier = MLPClassifier(fusion_dim, [fusion_h, fusion_h // 2])

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

        # ép h_g = 0 nếu không có graph
        h_g = h_g * has_graph.view(-1, 1)

        # logit branch
        gate_in = torch.cat([h_s, h_g], dim=-1)      # [B, d_s + d_g]
        scores = self.gate(gate_in)                  # [B, 2]

        # ép logit graph nhỏ nếu has_graph=0
        scores[:, 1] = scores[:, 1] + (has_graph - 1.0) * 1e4

        w = torch.softmax(scores, dim=-1)            # [B, 2]
        w_seq   = w[:, 0:1]
        w_graph = w[:, 1:2]

        h_fused = torch.cat([w_seq * h_s, w_graph * h_g], dim=-1)
        logits = self.classifier(h_fused)

        # không cần branch/features
        if not return_branch and not return_features:
            return logits

        branch_info = {
            "w_seq":    w_seq.detach().cpu(),
            "w_graph":  w_graph.detach().cpu(),
            "has_graph": has_graph.detach().cpu()
        }

        if return_branch and not return_features:
            return logits, branch_info
        if return_features and not return_branch:
            return logits, h_fused
        if return_branch and return_features:
            return logits, h_fused, branch_info

        return logits  # fallback


# ======================================================
# Adversarial DA: GRL + Domain Discriminator
# ======================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """
    D: phân biệt feature thuộc domain source (0) hay target (1).
    """
    def __init__(self, in_dim, hidden=128, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat, lambd=1.0):
        # GRL để đảo chiều gradient vào encoder
        feat_rev = grad_reverse(feat, lambd)
        out = self.net(feat_rev)
        return out.squeeze(1)


# ======================================================
# Dataset wrapper để thêm domain label
# ======================================================

class DomainSubset(Dataset):
    """
    Wrapper: (ds, indices, domain_labels) -> (graph, seq, label, domain_id)
    domain_id: 0 = source, 1 = target
    """
    def __init__(self, base_ds, indices, domain_labels):
        self.base_ds = base_ds
        self.indices = sorted(indices)
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        g, seq, y = self.base_ds[real_idx]
        d = self.domain_labels[real_idx]
        return g, seq, y, torch.tensor(float(d), dtype=torch.float32)


def collate_fn_da(batch):
    graphs, seqs, labels, domains = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(seqs),
        torch.stack(labels),
        torch.stack(domains),
    )


# ======================================================
# XAI helper cho 1 sample
# ======================================================

def explain_one_sample(model, ds, idx, device):
    model.eval()
    data, seq, label = ds[idx]
    g = Batch.from_data_list([data])

    g = g.to(device)
    seq = seq.unsqueeze(0).to(device)  # [1, L]
    label = label.item()

    with torch.no_grad():
        logits, br = model(g, seq, return_branch=True)
        prob = torch.sigmoid(logits)[0].item()

    w_seq = br["w_seq"][0].item()
    w_graph = br["w_graph"][0].item()
    has_graph = int(br["has_graph"][0].item())

    print(f"[Sample {idx}] label={label}, prob={prob:.4f}, pred={(prob>0.5):d}")
    print(f"[Branch] has_graph={has_graph}, "
          f"w_seq={w_seq:.3f}, w_graph={w_graph:.3f}")


# ======================================================
# Train / Eval
# ======================================================

def train_epoch_da(
    model,
    domain_disc,
    loader,
    cls_crit,
    dom_crit,
    opt_main,
    opt_disc,
    device,
    lambda_da=1.0,
    alpha_dom=0.1,
):
    """
    Train 1 epoch với Adversarial DA:
      L_total = L_cls + alpha_dom * L_domain

    Thống kê:
      - acc:     accuracy phân loại ransomware/benign
      - dom_acc: accuracy phân biệt domain (source/target) của D
    """
    model.train()
    domain_disc.train()
    total_loss = total_cls = total_dom = 0.0
    correct, total = 0, 0
    dom_correct, dom_total = 0, 0

    for g, seq, labs, doms in loader:
        g, seq, labs, doms = g.to(device), seq.to(device), labs.to(device), doms.to(device)

        opt_main.zero_grad()
        opt_disc.zero_grad()

        # 1) Forward encoder + classifier => logits, feat
        logits, feat = model(g, seq, return_features=True)

        # Align batch nếu mismatch
        b_log, b_lab = logits.size(0), labs.size(0)
        if b_log != b_lab:
            b = min(b_log, b_lab)
            logits = logits[:b]
            labs   = labs[:b]
            doms   = doms[:b]
            feat   = feat[:b]

        # 2) Classification loss
        cls_loss = cls_crit(logits, labs)

        # 3) Domain loss (adversarial nhờ GRL)
        dom_logits = domain_disc(feat, lambd=lambda_da)
        dom_loss = dom_crit(dom_logits, doms)

        # 4) Tổng loss
        loss = cls_loss + alpha_dom * dom_loss

        loss.backward()
        opt_main.step()
        opt_disc.step()

        # ====== Thống kê ======
        total_loss += loss.item() * labs.size(0)
        total_cls  += cls_loss.item() * labs.size(0)
        total_dom  += dom_loss.item() * labs.size(0)

        # acc phân loại ransomware/benign
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labs).sum().item()
        total   += labs.size(0)

        # acc domain (source/target) của D
        dom_pred = (torch.sigmoid(dom_logits) > 0.5).float()
        dom_correct += (dom_pred == doms).sum().item()
        dom_total   += doms.size(0)

    return {
        "loss":     total_loss / total,
        "cls_loss": total_cls  / total,
        "dom_loss": total_dom  / total,
        "acc":      correct    / total,
        "dom_acc":  dom_correct / max(dom_total, 1e-9),
    }


def train_epoch(model, loader, crit, opt, device):
    """
    Train thường (không DA) – baseline.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for g, seq, labs in loader:
        g, seq, labs = g.to(device), seq.to(device), labs.to(device)

        opt.zero_grad()
        logits = model(g, seq)

        if logits.size(0) != labs.size(0):
            b = min(logits.size(0), labs.size(0))
            logits = logits[:b]
            labs   = labs[:b]

        loss = crit(logits, labs)
        loss.backward()
        opt.step()

        total_loss += loss.item() * labs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labs).sum().item()
        total   += labs.size(0)

    return total_loss / total, correct / total


def eval_metrics(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for g, seq, labs in loader:
            g, seq, labs = g.to(device), seq.to(device), labs.to(device)
            logits = model(g, seq)

            if logits.size(0) != labs.size(0):
                b = min(logits.size(0), labs.size(0))
                logits = logits[:b]
                labs   = labs[:b]

            total_loss += crit(logits, labs).item() * labs.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.cpu().tolist())

    if len(set(all_labels)) < 2:
        return {
            'loss': total_loss,
            'acc': 0, 'tpr': 0, 'fpr': 0,
            'precision': 0, 'recall': 0, 'f1': 0
        }

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total = tp + tn + fp + fn

    return {
        'loss': total_loss / total,
        'acc': (tp + tn) / total,
        'tpr': tp / (tp + fn + 1e-9),
        'fpr': fp / (fp + tn + 1e-9),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1':  f1_score(all_labels, all_preds)
    }

# ======================================================
# Drift / PSI utilities (RANSOM-ONLY, YEAR concept drift)
# ======================================================

def _is_ransom(ds, idx: int) -> bool:
    return int(ds.samples[idx][2]) == 1

@torch.no_grad()
def collect_model_outputs_branch(model, ds, indices, device, batch_size=32):
    """
    Thu output cho PSI:
      - prob = sigmoid(logits)
      - w_seq, w_graph (từ gating)
    """
    if len(indices) == 0:
        return {"prob": np.array([]), "w_seq": np.array([]), "w_graph": np.array([])}

    loader = DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()
    probs, wseqs, wgraphs = [], [], []

    for g, seq, _ in loader:
        g = g.to(device)
        seq = seq.to(device)

        logits, br = model(g, seq, return_branch=True)
        p = torch.sigmoid(logits).detach().cpu().numpy()

        probs.append(p)
        wseqs.append(br["w_seq"].numpy().reshape(-1))
        wgraphs.append(br["w_graph"].numpy().reshape(-1))

    return {
        "prob": np.concatenate(probs, axis=0),
        "w_seq": np.concatenate(wseqs, axis=0),
        "w_graph": np.concatenate(wgraphs, axis=0),
    }

def make_bins_from_reference(ref_values, n_bins=10, strategy="quantile"):
    ref_values = np.asarray(ref_values)
    if ref_values.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    if strategy == "fixed":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(ref_values, qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def psi(expected, actual, bin_edges, eps=1e-6):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    e_counts, _ = np.histogram(expected, bins=bin_edges)
    a_counts, _ = np.histogram(actual, bins=bin_edges)

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def report_year_drift_psi_ransom_only(
    model, ds,
    tr_idx, val_idx, test_idx,
    domain_labels,
    device,
    batch_size=32,
    n_bins=10,
    bin_strategy_prob="quantile",
    bin_strategy_w="fixed",
):
    """
    Concept drift (YEAR) cho ransomware:
      Ref = (train+val) ∩ domain=0(2021) ∩ y=1
      Cur = test        ∩ domain=1(2025) ∩ y=1
    """
    ref_idx = [i for i in (tr_idx + val_idx) if domain_labels[i] == 0 and _is_ransom(ds, i)]
    cur_idx = [i for i in test_idx if domain_labels[i] == 1 and _is_ransom(ds, i)]

    print("\n=========== DRIFT REPORT (PSI) [RANSOMWARE YEAR] ===========")
    print(f"[SET] ref_ran={len(ref_idx)} (train+val,2021) | cur_ran={len(cur_idx)} (test,2025)")

    if len(ref_idx) == 0 or len(cur_idx) == 0:
        print("[ERROR] Thiếu ransomware ở ref hoặc cur => không tính PSI được.")
        print("===========================================================\n")
        return None

    ref_out = collect_model_outputs_branch(model, ds, ref_idx, device, batch_size=batch_size)
    cur_out = collect_model_outputs_branch(model, ds, cur_idx, device, batch_size=batch_size)

    edges_prob = make_bins_from_reference(ref_out["prob"], n_bins=n_bins, strategy=bin_strategy_prob)
    psi_prob = psi(ref_out["prob"], cur_out["prob"], edges_prob)

    edges_w = make_bins_from_reference(ref_out["w_graph"], n_bins=n_bins, strategy=bin_strategy_w)
    psi_wg = psi(ref_out["w_graph"], cur_out["w_graph"], edges_w)
    psi_ws = psi(ref_out["w_seq"],   cur_out["w_seq"],   edges_w)

    print(f"[PSI] PSI(prob)   ransom-only: {psi_prob:.6f}")
    print(f"[PSI] PSI(w_graph) ransom-only: {psi_wg:.6f}")
    print(f"[PSI] PSI(w_seq)   ransom-only: {psi_ws:.6f}")
    print("===========================================================\n")

    return {
        "psi_prob_ransom": psi_prob,
        "psi_w_graph_ransom": psi_wg,
        "psi_w_seq_ransom": psi_ws,
        "refN_ransom": len(ref_idx),
        "curN_ransom": len(cur_idx),
    }



# ======================================================
# Split từ metadata + domain label (source/target)
# ======================================================

def build_split_indices_from_metadata(ds, metadata_csv_path):
    """
    metadata.csv:
      - filename: 'report_xxx.json'
      - split: 'train' / 'val' / 'test'
      - domain: 'source' / 'target' (hoặc pre/old/2021 vs post/new/2025...)

    Trả về:
      - train_idx, val_idx, test_idx
      - domain_labels: list cùng độ dài ds.samples, mỗi phần tử là 0 (source) hoặc 1 (target)
    """
    split_map = {}   # base_name -> (split, dom_id)

    with open(metadata_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']            # ví dụ 'report_abc123.json'
            base = os.path.splitext(fname)[0]  # 'report_abc123'
            split = row['split_KB1'].strip().lower()

            dom_str = row.get('domain', 'source').strip().lower()
            if dom_str in ('source', 'src', 's', 'pre', 'old', '2021'):
                dom_id = 0
            elif dom_str in ('target', 'tgt', 't', 'post', 'new', '2025'):
                dom_id = 1
            else:
                dom_id = 0  # default source
                
            t25_level = row.get('target_2025_level', '').strip().lower()
            split_map[base] = (split, dom_id, t25_level)

    train_idx, val_idx, test_idx = [], [], []
    domain_labels = [0] * len(ds.samples)
    missing = 0

    for idx, (g_path, seq_path, lbl) in enumerate(ds.samples):
        base_g = os.path.splitext(os.path.basename(g_path))[0]
        val = split_map.get(base_g)

        if val is None:
            missing += 1
            continue

        split, dom_id, t25_level = val
        domain_labels[idx] = dom_id

        if split == "train":
            train_idx.append(idx)
        elif split == "val":
            val_idx.append(idx)
        elif split == "test":
            if t25_level == "early25" or dom_id == 0:
                test_idx.append(idx)
        else:
            missing += 1
            
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)
    
    print(f"[INFO] Split indices from metadata: "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
          f"missing_meta={missing}")
    return train_idx, val_idx, test_idx, domain_labels


# ======================================================
# Run function (GCN + xLSTM + Adversarial DA)
# ======================================================

def run_gcn_xlstm():
    # Graph paths
    benign_graph_dir = "/kaggle/input/2026benign-1633rans21/PYGdata_benign_2026_Rans21_1633/benign"
    ransomware_graph_dirs = [
        "/kaggle/input/2026benign-1633rans21/PYGdata_benign_2026_Rans21_1633/ransomware",
        "/kaggle/input/557rans25/pyg_data_DistilBERT_ransomware2025/ransomware"
    ]
    # Sequence paths
    seq_root = "/kaggle/input/seq-ids/seq_ids"
    # Vocab
    vocab_json_path = "/kaggle/input/seq-vocab/vocab_runtime.json"

    metadata_csv_path = "/kaggle/input/dataset-psi-split-year/metadata_KB1_with_target25_split.csv"

    # Load vocab
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"[INFO] Vocab size = {vocab_size}")

    seq_len_options = [1500]
    batch_size = 8
    lr = 1e-3
    epochs = 20
    patience = 5

    # Hyper-params cho Domain Adversarial
    lambda_da = 1.0   # độ mạnh GRL
    alpha_dom = 0.5   # trọng số domain loss trong tổng loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    for seq_len in seq_len_options:
        print(f"\n=== [SEQ_LEN={seq_len}] (Adversarial DA) ===")

        try:
            ds = MultiModalDatasetPT(
                benign_graph_dir,
                ransomware_graph_dirs,
                seq_root,
                max_seq_len=seq_len
            )
        except RuntimeError as e:
            print(f"[WARN] {e} → skip combo này.")
            return

        print(f"[INFO] Found {len(ds)} samples (before applying metadata splits)")
        if len(ds) < 10:
            print("[WARNING] Not enough samples. Skipping...")
            return

        # Split + domain label từ metadata
        tr_idx, val_idx, test_idx, domain_labels = build_split_indices_from_metadata(
            ds, metadata_csv_path
        )

        if min(len(tr_idx), len(val_idx), len(test_idx)) == 0:
            print("[ERROR] One of the splits is empty (train/val/test). "
                  "Check metadata.csv and file naming.")
            return

        tr_loader = DataLoader(
            DomainSubset(ds, tr_idx, domain_labels),
            batch_size,
            shuffle=True,
            collate_fn=collate_fn_da
        )
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            Subset(ds, test_idx),
            batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Lấy feature dim từ sample
        sample_g, _, _ = ds[0]
        in_feats = sample_g.x.size(1)
        print(f"[INFO] Node feature dim = {in_feats}")

        # Encoders + classifier
        graph_enc = GCNEncoder(in_feats).to(device)
        seq_enc   = xLSTMEncoder(vocab_size, seq_len=seq_len).to(device)
        model     = MultiModalClassifier(graph_enc, seq_enc).to(device)

        # Domain Discriminator trên fused feature
        domain_disc = DomainDiscriminator(model.fusion_dim).to(device)

        cls_crit = nn.BCEWithLogitsLoss()
        dom_crit = nn.BCEWithLogitsLoss()

        opt_main = torch.optim.Adam(model.parameters(), lr=lr)
        opt_disc = torch.optim.Adam(domain_disc.parameters(), lr=lr)

        best_f1 = 0.0
        no_improve = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            tr_stats = train_epoch_da(
                model,
                domain_disc,
                tr_loader,
                cls_crit,
                dom_crit,
                opt_main,
                opt_disc,
                device,
                lambda_da=lambda_da,
                alpha_dom=alpha_dom,
            )
            val_met = eval_metrics(model, val_loader, cls_crit, device)

            print(
                f"[Epoch {epoch}] "
                f"Train Loss={tr_stats['loss']:.6f} "
                f"(cls={tr_stats['cls_loss']:.6f}, dom={tr_stats['dom_loss']:.6f}), "
                f"Train Acc={tr_stats['acc']:.6f}, "
                f"Train DomAcc={tr_stats['dom_acc']:.6f} | "
                f"Val Loss={val_met['loss']:.6f}, Val Acc={val_met['acc']:.6f}, "
                f"Val F1={val_met['f1']:.6f}"
            )

            # Early stopping theo Val F1
            if val_met['f1'] > best_f1:
                best_f1 = val_met['f1']
                best_state = (
                    model.state_dict(),
                    opt_main.state_dict(),
                    domain_disc.state_dict(),
                    opt_disc.state_dict(),
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break

        # Load lại best state
        if best_state is not None:
            model.load_state_dict(best_state[0])
            opt_main.load_state_dict(best_state[1])
            domain_disc.load_state_dict(best_state[2])
            opt_disc.load_state_dict(best_state[3])

        # Đánh giá test như cũ
        test_met = eval_metrics(model, test_loader, cls_crit, device)
        print(
            f">>> TEST (Adversarial DA) => "
            f"Loss={test_met['loss']:.6f}, Acc={test_met['acc']:.6f}, "
            f"TPR={test_met['tpr']:.6f}, FPR={test_met['fpr']:.6f}, "
            f"Precision={test_met['precision']:.6f}, Recall={test_met['recall']:.6f}, "
            f"F1={test_met['f1']:.6f}"
        )

        # ===== PSI drift (ransom-only, year concept drift) =====
        drift_stats = report_year_drift_psi_ransom_only(
            model, ds,
            tr_idx, val_idx, test_idx,
            domain_labels,
            device=device,
            batch_size=32,
            n_bins=10,
            bin_strategy_prob="quantile",
            bin_strategy_w="fixed",
        )
        print("[DRIFT_STATS]", drift_stats)


if __name__ == '__main__':
    set_seed(42)
    run_gcn_xlstm()