import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
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

# -----------------------------------
# Utility
# -----------------------------------
def load_json(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)

# -----------------------------------
# Dataset
# -----------------------------------
class MultiModalDataset(Dataset):
    """
    Returns (PyG Data, seq_tensor, label).
    JSONs for xlstm at:
      - benign: /kaggle/input/json-atb-benign-507
      - ransomware: /kaggle/input/ransom-5xx-new/ransomware
    PTs for gcn at:
      - benign: /kaggle/input/pyg-data-distilbert/pyg_data_DistilBERT/benign
      - ransomware: /kaggle/input/pyg-data-distilbert/pyg_data_DistilBERT/ransomware
    """
    def __init__(self, json_root: str, pt_root: str, max_seq_len: int = 1500):
        self.samples = []
        self.vocab = {'<PAD>':0, '<UNK>':1}
        idx = 2
        self.max_len = max_seq_len
        # Define json and pt subfolders with labels
        mapping = [
            (os.path.join(json_root, 'json-atb-benign-507'),
             os.path.join(pt_root, 'benign'), 0),
            (os.path.join(json_root, 'ransom-5xx-new', 'ransomware'),
             os.path.join(pt_root, 'ransomware'), 1)
        ]
        for json_dir, pt_dir, label in mapping:
            if not os.path.isdir(json_dir) or not os.path.isdir(pt_dir):
                print(f"Warning: missing {json_dir} or {pt_dir}, skipping")
                continue
            for fname in os.listdir(json_dir):
                if not fname.endswith('.json'): continue
                sample_id = os.path.splitext(fname)[0]
                json_path = os.path.join(json_dir, fname)
                pt_path = os.path.join(pt_dir, f"{sample_id}.pt")
                if not os.path.isfile(pt_path):
                    continue
                feat = load_json(json_path)
                tokens = []
                # API calls
                for call in feat.get('api_call_sequence', [])[:1000]:
                    tokens.append(f"api:{call.get('api','')}")
                # behavior_summary
                for ft, vals in feat.get('behavior_summary', {}).items():
                    for v in vals:
                        tokens.append(f"feature:{ft}:{v}")
                # dropped_files
                for d in feat.get('dropped_files', []):
                    if isinstance(d, dict): tokens.append(f"dropped_file:{d.get('filepath','')}")
                    else: tokens.append(f"dropped_file:{d}")
                # signatures
                for sig in feat.get('signatures', []):
                    tokens.append(f"signature:{sig.get('name','')}")
                # processes
                for p in feat.get('processes', []):
                    tokens.append(f"process:{p.get('name','')}")
                # network
                for proto, entries in feat.get('network', {}).items():
                    for e in entries:
                        if isinstance(e, dict):
                            dst = e.get('dst') or e.get('dst_ip','')
                            port= e.get('dst_port') or e.get('port','')
                            tokens.append(f"network:{proto}:{dst}:{port}")
                        else:
                            tokens.append(f"network:{proto}:{e}")
                # update vocab
                for t in tokens:
                    if t not in self.vocab:
                        self.vocab[t] = idx; idx += 1
                self.samples.append((pt_path, tokens, label))
        print(f"Loaded {len(self.samples)} samples; Vocab size = {len(self.vocab)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pt_path, tokens, label = self.samples[i]
        data = torch.load(pt_path, weights_only=False)
        idxs = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(idxs) >= self.max_len:
            idxs = idxs[:self.max_len]
        else:
            idxs += [self.vocab['<PAD>']] * (self.max_len - len(idxs))
        seq = torch.tensor(idxs, dtype=torch.long)
        return data, seq, torch.tensor(label, dtype=torch.float32)

# collate function
from torch_geometric.data import Batch

def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    seqs = torch.stack(seqs, dim=0)
    labels = torch.stack(labels, dim=0)
    return batch_graph, seqs, labels

# -----------------------------------
# Encoders
# -----------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.bn1   = BatchNorm(hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.bn2   = BatchNorm(hidden_feats)
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)
        return global_mean_pool(x, batch)

class xLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, seq_len=1500, num_blocks=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=seq_len,
            num_blocks=num_blocks,
            embedding_dim=embed_dim,
            slstm_at=[0]  # position must be < num_blocks
        )
        self.xlstm = xLSTMBlockStack(cfg)
    def forward(self, seq):
        emb = self.embed(seq)
        out = self.xlstm(emb)
        return out.mean(dim=1)

# -----------------------------------
# MLP classifier
# -----------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.3):
        super().__init__()
        layers=[]
        dims=[input_dim]+hidden_dims
        for i in range(len(hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1],1))
        self.mlp = nn.Sequential(*layers)
    def forward(self,x): return self.mlp(x).squeeze(1)

# -----------------------------------
# Fusion model
# -----------------------------------
class MultiModalClassifier(nn.Module):
    def __init__(self, gcn_enc, xlstm_enc, fusion_hidden=128):
        super().__init__()
        self.gcn = gcn_enc
        self.xlstm = xlstm_enc
        fusion_dim = gcn_enc.conv2.out_channels + xlstm_enc.embed.embedding_dim
        self.classifier = MLPClassifier(fusion_dim, [fusion_hidden, fusion_hidden//2])
    def forward(self, graph_data, seq):
        x, edge_idx, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        g_emb = self.gcn(x, edge_idx, batch)
        s_emb = self.xlstm(seq)
        fused = torch.cat([g_emb, s_emb], dim=1)
        return self.classifier(fused)

# -----------------------------------
# Train/Eval
# -----------------------------------
def train_epoch(model, loader, crit, opt, device):
    model.train(); tloss=0; correct=0; total=0
    for graph, seq, labs in loader:
        graph, seq, labs = graph.to(device), seq.to(device), labs.to(device)
        opt.zero_grad(); logits = model(graph, seq)
        loss = crit(logits, labs); loss.backward(); opt.step()
        tloss += loss.item()*labs.size(0)
        preds = (torch.sigmoid(logits)>0.5).float()
        correct += (preds==labs).sum().item(); total += labs.size(0)
    return tloss/total, correct/total

def eval_epoch(model, loader, crit, device):
    model.eval(); tloss=0; correct=0; total=0
    with torch.no_grad():
        for graph, seq, labs in loader:
            graph, seq, labs = graph.to(device), seq.to(device), labs.to(device)
            logits = model(graph, seq)
            loss = crit(logits, labs)
            tloss += loss.item()*labs.size(0)
            preds = (torch.sigmoid(logits)>0.5).float()
            correct += (preds==labs).sum().item(); total += labs.size(0)
    return tloss/total, correct/total

# -----------------------------------
# Main
# -----------------------------------
def main():
    # Updated paths
    json_root = "/kaggle/input"
    pt_root   = "/kaggle/input/pyg-data-distilbert/pyg_data_DistilBERT"
    max_seq_len=1500; batch_size=8; lr=1e-3; epochs=20
    gcn_h=64; lstm_e=128; lstm_b=1; fusion_h=128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    ds = MultiModalDataset(json_root, pt_root, max_seq_len)
    train_n=int(0.8*len(ds)); test_n=len(ds)-train_n
    tr_ds, te_ds = random_split(ds, [train_n, test_n])
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    sample_graph, _, _ = ds[0]
    in_ch = sample_graph.x.size(1)
    gcn_enc   = GCNEncoder(in_ch, gcn_h).to(device)
    xlstm_enc = xLSTMEncoder(len(ds.vocab), lstm_e, max_seq_len, lstm_b).to(device)
    model     = MultiModalClassifier(gcn_enc, xlstm_enc, fusion_h).to(device)
    crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_losses, te_losses, te_accs = [], [], []
    best=0
    for ep in range(1, epochs+1):
        tl, ta = train_epoch(model, tr_ld, crit, opt, device)
        vl, va = eval_epoch(model, te_ld, crit, device)
        tr_losses.append(tl); te_losses.append(vl); te_accs.append(va)
        print(f"Epoch {ep} | Train: {tl:.4f}/{ta:.4f} | Val: {vl:.4f}/{va:.4f}")
        if va>best: best=va; torch.save(model.state_dict(), 'best_multimodal.pth')
    print(f"Best Val Acc: {best:.4f}")
    plt.figure(figsize=(8,5))
    plt.plot(tr_losses, label='Train Loss')
    plt.plot(te_losses, label='Val Loss')
    plt.plot(te_accs, label='Val Acc')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('multimodal_convergence.png', dpi=300)

if __name__=='__main__':
    main()
