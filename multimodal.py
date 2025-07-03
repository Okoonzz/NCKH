import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

# xLSTM imports
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig
)

# --------------------------
# Dataset with vocab caching
# --------------------------
class MultiModalDataset(Dataset):
    CACHE_FILE = '/kaggle/working/vocab.json'

    def __init__(self, json_root, pt_root, max_seq_len=1500):
        self.max_len = max_seq_len
        self.samples = []
        # Load or build vocab
        if os.path.isfile(self.CACHE_FILE):
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as vf:
                self.vocab = json.load(vf)
            idx = max(self.vocab.values()) + 1
            print(f"[INFO] Loaded vocab from cache: {self.CACHE_FILE} (size: {len(self.vocab)})")
        else:
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            idx = 2
            print("[INFO] Building vocab from scratch...")
        # Directories and labels
        mapping = [
            (os.path.join(json_root, 'json-atb-benign-507'), os.path.join(pt_root, 'benign'), 0),
            (os.path.join(json_root, 'ransom-5xx-new', 'ransomware'), os.path.join(pt_root, 'ransomware'), 1)
        ]
        for jdir, pdir, label in mapping:
            if not os.path.isdir(jdir) or not os.path.isdir(pdir): continue
            for fname in os.listdir(jdir):
                if not fname.endswith('.json'): continue
                sid = os.path.splitext(fname)[0]
                jpath = os.path.join(jdir, fname)
                ppath = os.path.join(pdir, f"{sid}.pt")
                if not os.path.isfile(ppath): continue
                feat = self._load_json(jpath)
                toks = self._extract_tokens(feat)
                if not os.path.isfile(self.CACHE_FILE):
                    for t in toks:
                        if t not in self.vocab:
                            self.vocab[t] = idx; idx += 1
                self.samples.append((ppath, toks, label))
        # Save vocab cache
        if not os.path.isfile(self.CACHE_FILE):
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as vf:
                json.dump(self.vocab, vf, ensure_ascii=False, indent=2)

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)

    def _extract_tokens(self, feat):
        toks = []
        for call in feat.get('api_call_sequence', [])[:1000]:
            toks.append(f"api:{call.get('api','')}")
        for ft, vals in feat.get('behavior_summary', {}).items():
            for v in vals: toks.append(f"feature:{ft}:{v}")
        for d in feat.get('dropped_files', []):
            toks.append(f"dropped_file:{d if not isinstance(d,dict) else d.get('filepath','')}" )
        for sig in feat.get('signatures', []): toks.append(f"signature:{sig.get('name','')}" )
        for p in feat.get('processes', []): toks.append(f"process:{p.get('name','')}" )
        for proto, ents in feat.get('network', {}).items():
            for e in ents:
                if isinstance(e, dict):
                    dst = e.get('dst') or e.get('dst_ip','')
                    port = e.get('dst_port') or e.get('port','')
                    toks.append(f"network:{proto}:{dst}:{port}")
                else:
                    toks.append(f"network:{proto}:{e}")
        return toks

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ppath, toks, label = self.samples[i]
        data = torch.load(ppath, weights_only=False)
        idxs = [self.vocab.get(t, self.vocab['<UNK>']) for t in toks]
        if len(idxs) < self.max_len:
            idxs += [self.vocab['<PAD>']] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        seq = torch.tensor(idxs, dtype=torch.long)
        return data, seq, torch.tensor(label, dtype=torch.float32)

# --------------------------
# Collate function
# --------------------------
def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(seqs), torch.stack(labels)

# --------------------------
# Encoders
# --------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = BatchNorm(hidden)
        self.drop = drop
        self.output_dim = hidden
    def forward(self, x, ei, batch):
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        return global_mean_pool(x, batch)

class GATEncoder(GCNEncoder):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__(in_feats, hidden, drop)
        self.conv1 = GATConv(in_feats, hidden, heads=4, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
        self.output_dim = hidden

class SageEncoder(GCNEncoder):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__(in_feats, hidden, drop)
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.output_dim = hidden

class GINEncoder(nn.Module):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_feats, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gin1 = GINConv(nn1)
        self.bn1 = BatchNorm(hidden)
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gin2 = GINConv(nn2)
        self.bn2 = BatchNorm(hidden)
        self.drop = drop
        self.output_dim = hidden
    def forward(self, x, ei, batch):
        x = F.relu(self.bn1(self.gin1(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        x = F.relu(self.bn2(self.gin2(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        return global_mean_pool(x, batch)

class xLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed=128, seq_len=1500, blocks=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=seq_len,
            num_blocks=blocks,
            embedding_dim=embed,
            slstm_at=[0]
        )
        self.xlstm = xLSTMBlockStack(cfg)
        self.output_dim = embed
    def forward(self, seq):
        out = self.xlstm(self.embed(seq))
        return out.mean(dim=1)

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=128, layers=1, bidir=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, bidirectional=bidir)
        self.output_dim = hidden * (2 if bidir else 1)
    def forward(self, seq):
        emb = self.embed(seq)
        _, (hn, _) = self.lstm(emb)
        return torch.cat([hn[-2], hn[-1]], dim=1) if self.lstm.bidirectional else hn[-1]

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hiddens=[128,64], drop=0.3):
        super().__init__()
        layers = []
        dims = [in_dim] + hiddens
        for i in range(len(hiddens)):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(drop)])
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x).squeeze(1)

class MultiModalClassifier(nn.Module):
    def __init__(self, graph_enc, seq_enc, fusion_h=128):
        super().__init__()
        self.graph_enc = graph_enc
        self.seq_enc = seq_enc
        fusion_dim = graph_enc.output_dim + seq_enc.output_dim
        self.classifier = MLPClassifier(fusion_dim, [fusion_h, fusion_h//2])
    def forward(self, g, seq):
        g_emb = self.graph_enc(g.x, g.edge_index, g.batch)
        s_emb = self.seq_enc(seq)
        return self.classifier(torch.cat([g_emb, s_emb], dim=1))

# Training/evaluation

def train_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for g, seq, labs in loader:
        g, seq, labs = g.to(device), seq.to(device), labs.to(device)
        opt.zero_grad()
        logits = model(g, seq)
        loss = crit(logits, labs)
        loss.backward()
        opt.step()
        total_loss += loss.item() * labs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labs).sum().item()
        total += labs.size(0)
    return total_loss/total, correct/total


def eval_metrics(model, loader, crit, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for g, seq, labs in loader:
            g, seq, labs = g.to(device), seq.to(device), labs.to(device)
            logits = model(g, seq)
            total_loss += crit(logits, labs).item() * labs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.cpu().tolist())
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total = tp + tn + fp + fn
    return {
        'loss': total_loss/total,
        'acc': (tp+tn)/total,
        'tpr': tp/(tp+fn+1e-9),
        'fpr': fp/(fp+tn+1e-9),
        'f1': f1_score(all_labels, all_preds)
    }

# Main without cross-validation

def main():
    json_root = "/kaggle/input"
    pt_root = "/kaggle/input/pyg-data-distilbert/pyg_data_DistilBERT"
    max_seq_len = 1500
    batch_size = 8
    lr = 1e-3
    epochs = 20
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset and stratified split
    ds = MultiModalDataset(json_root, pt_root, max_seq_len)
    labels = [lbl for _,_,lbl in ds.samples]
    # Split train+val vs test
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(outer.split(range(len(ds)), labels))
    # Split train vs val
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    y_tv = [labels[i] for i in train_val_idx]
    tr_idx_rel, val_idx_rel = next(inner.split(train_val_idx, y_tv))
    tr_idx = [train_val_idx[i] for i in tr_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]

    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size, shuffle=False, collate_fn=collate_fn)

    graph_models = {'gcn': GCNEncoder, 'gat': GATEncoder, 'sage': SageEncoder, 'gin': GINEncoder}
    seq_models = {'xlstm': xLSTMEncoder, 'lstm': LSTMEncoder}

    for g_name, g_cls in graph_models.items():
        for s_name, s_cls in seq_models.items():
            print(f"Running {g_name.upper()} + {s_name.upper()}")
            # Model init
            sample_g, _, _ = ds[0]
            in_feats = sample_g.x.size(1)
            graph_enc = g_cls(in_feats).to(device)
            if s_name == 'xlstm':
                seq_enc = s_cls(len(ds.vocab), seq_len=max_seq_len).to(device)
            else:
                seq_enc = s_cls(len(ds.vocab)).to(device)
            model = MultiModalClassifier(graph_enc, seq_enc).to(device)
            crit = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            # Training with early stopping
            best_f1 = 0
            no_improve = 0
            best_state = None
            for epoch in range(1, epochs+1):
                tr_loss, tr_acc = train_epoch(model, tr_loader, crit, opt, device)
                val_met = eval_metrics(model, val_loader, crit, device)
                print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
                      f"Val Loss: {val_met['loss']:.4f}, Val Acc: {val_met['acc']:.4f}, F1: {val_met['f1']:.4f}")
                if val_met['f1'] > best_f1:
                    best_f1 = val_met['f1']
                    best_state = (model.state_dict(), opt.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            # Load best and evaluate on test
            model.load_state_dict(best_state[0])
            opt.load_state_dict(best_state[1])
            test_met = eval_metrics(model, test_loader, crit, device)
            print(f"Test Results -> Loss: {test_met['loss']:.4f}, Acc: {test_met['acc']:.4f}, "
                  f"TPR: {test_met['tpr']:.4f}, FPR: {test_met['fpr']:.4f}, F1: {test_met['f1']:.4f}\n")

if __name__ == '__main__':
    main()
