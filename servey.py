import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# xLSTM
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig
)

class MultiModalDataset(Dataset):
    def __init__(self, json_root, pt_root, max_seq_len=1500):
        self.max_len = max_seq_len
        self.samples = []
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
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
                for t in toks:
                    if t not in self.vocab:
                        self.vocab[t] = idx; idx += 1
                self.samples.append((ppath, toks, label))

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

def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(seqs), torch.stack(labels)

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
    if len(set(all_labels)) < 2:
        return {'loss': total_loss, 'acc': 0, 'tpr': 0, 'fpr': 0, 'f1': 0}
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total = tp + tn + fp + fn
    return {
        'loss': total_loss/total,
        'acc': (tp+tn)/total,
        'tpr': tp/(tp+fn+1e-9),
        'fpr': fp/(fp+tn+1e-9),
        'f1': f1_score(all_labels, all_preds)
    }

def survey_gcn_xlstm():
    json_root = "/kaggle/input"
    api_options = [500, 1000, 1500]
    seq_len_options = [500, 1000, 1500]
    base_pt_root = "/kaggle/input"
    batch_size = 8
    lr = 1e-3
    epochs = 20
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for api_num in api_options:
        pt_root = os.path.join(base_pt_root, f"{api_num}-final/{api_num}")
        for seq_len in seq_len_options:
            print(f"\n=== [API={api_num}] [SEQ_LEN={seq_len}] ===")
            ds = MultiModalDataset(json_root, pt_root, max_seq_len=seq_len)
            print(f"[INFO] Found {len(ds)} samples")
            if len(ds) < 10:
                print("[WARNING] Not enough samples. Skipping...")
                continue
            labels = [lbl for _,_,lbl in ds.samples]
            outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            train_val_idx, test_idx = next(outer.split(range(len(ds)), labels))
            inner = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
            y_tv = [labels[i] for i in train_val_idx]
            tr_idx_rel, val_idx_rel = next(inner.split(train_val_idx, y_tv))
            tr_idx = [train_val_idx[i] for i in tr_idx_rel]
            val_idx = [train_val_idx[i] for i in val_idx_rel]

            tr_loader = DataLoader(Subset(ds, tr_idx), batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(Subset(ds, val_idx), batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(Subset(ds, test_idx), batch_size, shuffle=False, collate_fn=collate_fn)

            sample_g, _, _ = ds[0]
            in_feats = sample_g.x.size(1)
            graph_enc = GCNEncoder(in_feats).to(device)
            seq_enc = xLSTMEncoder(len(ds.vocab), seq_len=seq_len).to(device)
            model = MultiModalClassifier(graph_enc, seq_enc).to(device)
            crit = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            best_f1 = 0; no_improve = 0; best_state = None
            for epoch in range(1, epochs+1):
                tr_loss, tr_acc = train_epoch(model, tr_loader, crit, opt, device)
                val_met = eval_metrics(model, val_loader, crit, device)
                print(f"[Epoch {epoch}] Train Loss={tr_loss:.4f}, Acc={tr_acc:.4f} | Val F1={val_met['f1']:.4f}")
                if val_met['f1'] > best_f1:
                    best_f1 = val_met['f1']
                    best_state = (model.state_dict(), opt.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("Early stopping.")
                        break

            if best_state:
                model.load_state_dict(best_state[0])
                opt.load_state_dict(best_state[1])
            test_met = eval_metrics(model, test_loader, crit, device)
            print(f">>> TEST [API={api_num}] [SEQ_LEN={seq_len}] => "
                  f"Loss={test_met['loss']:.4f}, Acc={test_met['acc']:.4f}, "
                  f"TPR={test_met['tpr']:.4f}, FPR={test_met['fpr']:.4f}, F1={test_met['f1']:.4f}")

if __name__ == '__main__':
    survey_gcn_xlstm()
