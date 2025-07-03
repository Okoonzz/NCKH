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

# ───────────────────────────────────────────
# xLSTM imports
# ───────────────────────────────────────────
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig
)

# ───────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────
class MultiModalDataset(Dataset):
    CACHE_FILE = 'vocab.json'

    def __init__(self, json_root, pt_root, max_seq_len=1500):
        self.max_len = max_seq_len
        self.samples = []

        # vocab
        if os.path.isfile(self.CACHE_FILE):
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            idx = max(self.vocab.values()) + 1
            print(f"[INFO] Loaded vocab ({len(self.vocab)} tokens)")
        else:
            self.vocab, idx = {'<PAD>': 0, '<UNK>': 1}, 2
            print("[INFO] Building vocab from scratch…")

        # iterate folders
        mapping = [
            (os.path.join(json_root, 'json-atb-benign-507'),
             os.path.join(pt_root,  'benign'),      0),
            (os.path.join(json_root, 'ransom-5xx-new', 'ransomware'),
             os.path.join(pt_root,  'ransomware'),  1)
        ]
        for jdir, pdir, label in mapping:
            if not (os.path.isdir(jdir) and os.path.isdir(pdir)):
                continue
            for fname in os.listdir(jdir):
                if not fname.endswith('.json'):
                    continue
                sid   = os.path.splitext(fname)[0]
                jpath = os.path.join(jdir, fname)
                ppath = os.path.join(pdir, f"{sid}.pt")
                if not os.path.isfile(ppath):
                    continue

                feat = self._load_json(jpath)
                toks = self._extract_tokens(feat)

                if not os.path.isfile(self.CACHE_FILE):
                    for t in toks:
                        if t not in self.vocab:
                            self.vocab[t] = idx
                            idx += 1

                self.samples.append((ppath, toks, label))

        if not os.path.isfile(self.CACHE_FILE):
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_json(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)

    def _extract_tokens(self, feat):
        toks = []
        for call in feat.get('api_call_sequence', [])[:1000]:
            toks.append(f"api:{call.get('api','')}")
        for ft, vals in feat.get('behavior_summary', {}).items():
            toks += [f"feature:{ft}:{v}" for v in vals]
        for d in feat.get('dropped_files', []):
            toks.append(f"dropped:{d if not isinstance(d,dict) else d.get('filepath','')}")
        toks += [f"sig:{s.get('name','')}" for s in feat.get('signatures', [])]
        toks += [f"proc:{p.get('name','')}" for p in feat.get('processes', [])]
        for proto, ents in feat.get('network', {}).items():
            for e in ents:
                if isinstance(e, dict):
                    dst  = e.get('dst') or e.get('dst_ip','')
                    port = e.get('dst_port') or e.get('port','')
                    toks.append(f"net:{proto}:{dst}:{port}")
                else:
                    toks.append(f"net:{proto}:{e}")
        return toks

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ppath, toks, label = self.samples[i]
        graph = torch.load(ppath, weights_only=False)

        idxs = [self.vocab.get(t, 1) for t in toks]
        idxs = idxs[:self.max_len] + [0] * max(0, self.max_len - len(idxs))
        seq  = torch.tensor(idxs, dtype=torch.long)
        return graph, seq, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(seqs), torch.stack(labels)

# ───────────────────────────────────────────
# Encoders
# ───────────────────────────────────────────
class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2   = BatchNorm(hidden)
        self.drop  = drop
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

class SageEncoder(GCNEncoder):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__(in_feats, hidden, drop)
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

class GINEncoder(nn.Module):
    def __init__(self, in_feats, hidden=64, drop=0.3):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_feats, hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden))
        self.g1 = GINConv(nn1)
        self.g2 = GINConv(nn2)
        self.bn1 = BatchNorm(hidden)
        self.bn2 = BatchNorm(hidden)
        self.drop = drop
        self.output_dim = hidden

    def forward(self, x, ei, batch):
        x = F.relu(self.bn1(self.g1(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        x = F.relu(self.bn2(self.g2(x, ei)))
        x = F.dropout(x, self.drop, training=self.training)
        return global_mean_pool(x, batch)

class xLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed=128, seq_len=1500):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4,
                                       qkv_proj_blocksize=4, num_heads=4)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4,
                                       conv1d_kernel_size=4,
                                       bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=seq_len,
            num_blocks=1,
            embedding_dim=embed,
            slstm_at=[0]
        )
        self.xlstm = xLSTMBlockStack(cfg)
        self.output_dim = embed

    def forward(self, seq):
        return self.xlstm(self.embed(seq)).mean(dim=1)

class LSTMEncoder(nn.Module):          # Giữ lại để dùng cho các combo khác
    def __init__(self, vocab_size, embed=128, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm  = nn.LSTM(embed, hidden, batch_first=True)
        self.output_dim = hidden

    def forward(self, seq):
        _, (hn, _) = self.lstm(self.embed(seq))
        return hn.squeeze(0)

# ───────────────────────────────────────────
# Classifiers & wrappers
# ───────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hiddens=[128, 64], drop=0.3):
        super().__init__()
        dims, layers = [in_dim] + hiddens, []
        for i in range(len(hiddens)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(drop)]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x): return self.mlp(x).squeeze(1)

class GraphOnly(nn.Module):
    def __init__(self, enc): super().__init__(); self.enc=enc; self.fc=MLPClassifier(enc.output_dim)
    def forward(self, g,s): return self.fc(self.enc(g.x,g.edge_index,g.batch))

class SeqOnly(nn.Module):
    def __init__(self, enc): super().__init__(); self.enc=enc; self.fc=MLPClassifier(enc.output_dim)
    def forward(self, g,s): return self.fc(self.enc(s))

class MultiModal(nn.Module):
    def __init__(self, genc, senc, hid=128):
        super().__init__()
        self.genc, self.senc = genc, senc
        self.fc = MLPClassifier(genc.output_dim + senc.output_dim,
                                [hid, hid//2])
    def forward(self, g,s):
        return self.fc(torch.cat([self.genc(g.x,g.edge_index,g.batch),
                                  self.senc(s)],1))

# ───────────────────────────────────────────
# Train & eval
# ───────────────────────────────────────────
def train_epoch(model, loader, crit, opt, dev):
    model.train(); tot, ok, n = 0,0,0
    for g,s,l in loader:
        g,s,l = g.to(dev),s.to(dev),l.to(dev)
        opt.zero_grad()
        logit = model(g,s); loss = crit(logit,l); loss.backward(); opt.step()
        tot += loss.item()*l.size(0)
        ok  += ((torch.sigmoid(logit)>.5)==l).sum().item(); n += l.size(0)
    return tot/n, ok/n

def metrics(model, loader, crit, dev):
    model.eval(); tot, p, y = 0,[],[]
    with torch.no_grad():
        for g,s,l in loader:
            g,s,l = g.to(dev),s.to(dev),l.to(dev)
            logit = model(g,s); tot += crit(logit,l).item()*l.size(0)
            p+= (torch.sigmoid(logit)>.5).float().cpu().tolist(); y+=l.cpu().tolist()
    tn,fp,fn,tp = confusion_matrix(y,p).ravel(); n=tp+tn+fp+fn
    return {'loss':tot/n,'acc':(tp+tn)/n,'tpr':tp/(tp+fn+1e-9),
            'fpr':fp/(fp+tn+1e-9),'f1':f1_score(y,p)}

def run(name, model, tl, vl, te, lr, ep, patience, dev, ckpt=None):
    crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad=0,0; best_state=None
    for i in range(1,ep+1):
        tloss,tacc = train_epoch(model,tl,crit,opt,dev)
        v = metrics(model,vl,crit,dev)
        print(f"[{name}] Ep{i:02d} | TrainL {tloss:.4f} A {tacc:.4f} "
              f"| ValL {v['loss']:.4f} A {v['acc']:.4f} F1 {v['f1']:.4f}")
        if v['f1']>best:
            best=v['f1']; best_state=model.state_dict(); bad=0
            if ckpt: torch.save(best_state, ckpt)
        else:
            bad+=1
            if bad>=patience: print(f"[{name}] Early stop"); break
    model.load_state_dict(best_state)
    t = metrics(model,te,crit,dev)
    print(f"[{name}] TEST → L {t['loss']:.4f} A {t['acc']:.4f} "
          f"TPR {t['tpr']:.4f} FPR {t['fpr']:.4f} F1 {t['f1']:.4f}")
    if ckpt: print(f"[{name}] Saved best to {ckpt}\n")

# ───────────────────────────────────────────
# Main
# ───────────────────────────────────────────
def main():
    json_root   = "/kaggle/input"
    pt_root     = "/kaggle/input/1500-final/1500" # Chỉnh API tại đây
    max_len     = 1500
    bs, lr, ep, patience = 8, 1e-3, 20, 5
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MultiModalDataset(json_root, pt_root, max_len)
    labels = [l for _,_,l in ds.samples]

    outer = StratifiedShuffleSplit(1, test_size=.15, random_state=42)
    tv_idx, test_idx = next(outer.split(range(len(ds)), labels))
    inner = StratifiedShuffleSplit(1, test_size=.17647, random_state=42)
    y_tv = [labels[i] for i in tv_idx]
    tr_rel, va_rel = next(inner.split(tv_idx, y_tv))
    tr_idx = [tv_idx[i] for i in tr_rel]; va_idx=[tv_idx[i] for i in va_rel]

    tl = DataLoader(Subset(ds,tr_idx), bs, True,  collate_fn)
    vl = DataLoader(Subset(ds,va_idx),bs, False, collate_fn)
    te = DataLoader(Subset(ds,test_idx),bs, False, collate_fn)

    # GCN-only
    gfeat = ds[0][0].x.size(1)
    run("GCN",
        GraphOnly(GCNEncoder(gfeat).to(dev)).to(dev),
        tl,vl,te, lr,ep,patience,dev)

    # xLSTM-only
    run("xLSTM",
        SeqOnly(xLSTMEncoder(len(ds.vocab),seq_len=max_len).to(dev)).to(dev),
        tl,vl,te, lr,ep,patience,dev)

    # combos
    g_encoders = {'gcn': GCNEncoder, 'gat': GATEncoder, 'sage': SageEncoder, 'gin': GINEncoder}
    s_encoders = {'xlstm': xLSTMEncoder, 'lstm': LSTMEncoder}

    for gn,gc in g_encoders.items():
        for sn,sc in s_encoders.items():
            print()
            genc = gc(gfeat).to(dev)
            senc = sc(len(ds.vocab), seq_len=max_len) if sn=='xlstm' \
                   else sc(len(ds.vocab))
            senc = senc.to(dev)
            ckpt = 'best_gcn_xlstm.pt' if gn=='gcn' and sn=='xlstm' else None
            run(f"{gn.upper()}+{sn.upper()}",
                MultiModal(genc,senc).to(dev),
                tl,vl,te, lr,ep,patience,dev, ckpt)

if __name__ == '__main__':
    main()
