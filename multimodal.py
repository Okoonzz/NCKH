import os, json, torch, matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig
)

# ───────────────────────────────────────────────────────────────
# 1. Dataset ----------------------------------------------------
# ───────────────────────────────────────────────────────────────
def load_json(p):
    with open(p,'r',encoding='utf-8',errors='ignore') as f: return json.load(f)

class MultiModalDataset(Dataset):
    def __init__(self,json_root,pt_root,max_len=1500):
        self.samples=[]; self.vocab={'<PAD>':0,'<UNK>':1}; idx=2; self.max_len=max_len
        mapping=[(os.path.join(json_root,'json-atb-benign-507'),
                  os.path.join(pt_root,'benign'),0),
                 (os.path.join(json_root,'ransom-5xx-new','ransomware'),
                  os.path.join(pt_root,'ransomware'),1)]
        for jdir,pdir,lbl in mapping:
            if not (os.path.isdir(jdir) and os.path.isdir(pdir)): continue
            for fn in os.listdir(jdir):
                if not fn.endswith('.json'): continue
                sid=fn[:-5]; jpath=os.path.join(jdir,fn); pt=os.path.join(pdir,f'{sid}.pt')
                if not os.path.isfile(pt): continue
                feat=load_json(jpath); toks=[]
                toks += [f"api:{c.get('api','')}" for c in feat.get('api_call_sequence',[])[:1000]]
                for ft,vals in feat.get('behavior_summary',{}).items(): toks+=[f"feature:{ft}:{v}" for v in vals]
                for d in feat.get('dropped_files',[]): toks.append(f"dropped_file:{d if not isinstance(d,dict) else d.get('filepath','')}")
                toks += [f"signature:{s.get('name','')}" for s in feat.get('signatures',[])]
                toks += [f"process:{p.get('name','')}"   for p in feat.get('processes',[])]
                for proto,es in feat.get('network',{}).items():
                    for e in es:
                        if isinstance(e,dict):
                            toks.append(f"network:{proto}:{e.get('dst') or e.get('dst_ip','')}:{e.get('dst_port') or e.get('port','')}")
                        else: toks.append(f"network:{proto}:{e}")
                for t in toks:
                    if t not in self.vocab: self.vocab[t]=idx; idx+=1
                self.samples.append((pt,toks,lbl))
        print(f"Dataset: {len(self.samples)} samples | Vocab={len(self.vocab)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        pt,toks,lbl=self.samples[i]
        g=torch.load(pt,weights_only=False)
        idxs=[self.vocab.get(t,self.vocab['<UNK>']) for t in toks]
        idxs = (idxs + [0]*self.max_len)[:self.max_len]
        return g, torch.tensor(idxs), torch.tensor(lbl,dtype=torch.float32)

def collate_fn(batch):
    gs,seqs,labs=zip(*batch)
    return Batch.from_data_list(gs), torch.stack(seqs), torch.stack(labs)

# ───────────────────────────────────────────────────────────────
# 2. Encoders & Classifier --------------------------------------
# ───────────────────────────────────────────────────────────────
class GCNEncoder(nn.Module):
    def __init__(self,d_in,hid=64,drop=0.3):
        super().__init__()
        self.c1, self.b1 = GCNConv(d_in,hid), BatchNorm(hid)
        self.c2, self.b2 = GCNConv(hid,hid),  BatchNorm(hid)
        self.drop=drop
    def forward(self,x,e,b):
        x=F.relu(self.b1(self.c1(x,e))); x=F.dropout(x,self.drop,self.training)
        x=F.relu(self.b2(self.c2(x,e))); x=F.dropout(x,self.drop,self.training)
        return global_mean_pool(x,b)

class xLSTMEncoder(nn.Module):
    def __init__(self,vocab,emb=128,seq_len=1500,blocks=1):
        super().__init__()
        self.embed=nn.Embedding(vocab,emb,padding_idx=0)
        cfg=xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4,qkv_proj_blocksize=4,num_heads=4)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend='vanilla',num_heads=4,conv1d_kernel_size=4,bias_init='powerlaw_blockdependent'),
                feedforward=FeedForwardConfig(proj_factor=1.3,act_fn='gelu')),
            context_length=seq_len,num_blocks=blocks,embedding_dim=emb,slstm_at=[0])
        self.core=xLSTMBlockStack(cfg)
    def forward(self,seq): return self.core(self.embed(seq)).mean(dim=1)

class MLPClassifier(nn.Module):
    def __init__(self,d_in,hidden=[128,64],drop=0.3):
        super().__init__()
        layers=[]; dims=[d_in]+hidden
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i],dims[i+1]), nn.ReLU(), nn.Dropout(drop)]
        layers.append(nn.Linear(dims[-1],1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(1)

class MultiModalClassifier(nn.Module):
    def __init__(self,genc,senc):
        super().__init__()
        self.g,self.s = genc,senc
        fused_dim = genc.c2.out_channels + senc.embed.embedding_dim
        self.clf  = MLPClassifier(fused_dim)
    def forward(self,g,seq):
        return self.clf(torch.cat([self.g(g.x,g.edge_index,g.batch), self.s(seq)],dim=1))

# ───────────────────────────────────────────────────────────────
# 3. Metric helpers --------------------------------------------
# ───────────────────────────────────────────────────────────────
def batch_accuracy(logits,labels):
    return ((torch.sigmoid(logits)>0.5).float()==labels).sum().item() / labels.size(0)

def train_epoch(model,loader,crit,opt,dev):
    model.train(); tl=ta=0; n=0
    for g,seq,l in loader:
        g,seq,l=g.to(dev),seq.to(dev),l.to(dev); n+=l.size(0)
        opt.zero_grad(); logits=model(g,seq); loss=crit(logits,l); loss.backward(); opt.step()
        tl += loss.item()*l.size(0); ta += batch_accuracy(logits,l)*l.size(0)
    return tl/n, ta/n

def eval_epoch(model,loader,crit,dev):
    model.eval(); tl=ta=0; n=0
    with torch.no_grad():
        for g,seq,l in loader:
            g,seq,l=g.to(dev),seq.to(dev),l.to(dev); n+=l.size(0)
            logits=model(g,seq); loss=crit(logits,l)
            tl += loss.item()*l.size(0); ta += batch_accuracy(logits,l)*l.size(0)
    return tl/n, ta/n

def f1_val(model,loader,dev):
    model.eval(); allp,alll=[],[]
    with torch.no_grad():
        for g,seq,l in loader:
            g,seq=g.to(dev),seq.to(dev)
            allp.extend((torch.sigmoid(model(g,seq))>0.5).cpu().tolist())
            alll.extend(l.tolist())
    return f1_score(alll,allp)

def test_metrics(model,loader,crit,dev):
    model.eval(); tot=0; allp,alll=[],[]
    with torch.no_grad():
        for g,seq,l in loader:
            g,seq,l=g.to(dev),seq.to(dev),l.to(dev)
            tot+=crit(model(g,seq),l).item()*l.size(0)
            allp+=(torch.sigmoid(model(g,seq))>0.5).cpu().tolist()
            alll+=l.cpu().tolist()
    tn,fp,fn,tp=confusion_matrix(alll,allp).ravel(); t=tp+tn+fp+fn
    return {'loss':tot/t,'acc':(tp+tn)/t,'tpr':tp/(tp+fn+1e-9),
            'fpr':fp/(fp+tn+1e-9),'f1':f1_score(alll,allp)}

# ───────────────────────────────────────────────────────────────
# 4. Main -------------------------------------------------------
# ───────────────────────────────────────────────────────────────
def main():
    json_root="/kaggle/input"
    pt_root  ="/kaggle/input/1000-final/1000"
    bs=8; lr=1e-3; max_len=1500; epochs=20; patience=5
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=MultiModalDataset(json_root,pt_root,max_len)
    if len(ds)==0: raise RuntimeError("Empty dataset!")

    # stratified 70/15/15
    y=[lbl for *_,lbl in ds.samples]
    ss=StratifiedShuffleSplit(n_splits=1,test_size=0.30,random_state=42)
    tr_idx,temp_idx = next(ss.split(range(len(ds)),y))
    y_temp=[y[i] for i in temp_idx]
    ss2=StratifiedShuffleSplit(n_splits=1,test_size=0.50,random_state=42)
    va_rel,te_rel = next(ss2.split(temp_idx,y_temp))
    va_idx=[temp_idx[i] for i in va_rel]; te_idx=[temp_idx[i] for i in te_rel]

    tr_ld=DataLoader(Subset(ds,tr_idx),bs,True,collate_fn=collate_fn)
    va_ld=DataLoader(Subset(ds,va_idx),bs,False,collate_fn=collate_fn)
    te_ld=DataLoader(Subset(ds,te_idx),bs,False,collate_fn=collate_fn)

    g_dim=ds[0][0].x.size(1)
    model=MultiModalClassifier(GCNEncoder(g_dim).to(dev),
                               xLSTMEncoder(len(ds.vocab)).to(dev)).to(dev)
    crit=nn.BCEWithLogitsLoss(); opt=torch.optim.Adam(model.parameters(),lr=lr)

    # tracking lists
    tr_loss_hist, va_loss_hist, va_f1_hist = [], [], []
    best_f1, no_imp = 0, 0

    for ep in range(1,epochs+1):
        tr_l,tr_a = train_epoch(model,tr_ld,crit,opt,dev)
        va_l,va_a = eval_epoch(model,va_ld,crit,dev)
        va_f       = f1_val(model,va_ld,dev)
        tr_loss_hist.append(tr_l); va_loss_hist.append(va_l); va_f1_hist.append(va_f)

        print(f"Ep{ep:02d} | TrL {tr_l:.4f} | VaL {va_l:.4f} | VaF1 {va_f:.4f}")

        if va_f>best_f1:                           # improvement
            best_f1=va_f; no_imp=0
            torch.save(model.state_dict(),'best_multimodal.pth')
        else:
            no_imp+=1
        if no_imp>=patience:
            print("Early-stopping triggered.")
            break

    # ── load & test
    model.load_state_dict(torch.load('best_multimodal.pth'))
    tm=test_metrics(model,te_ld,crit,dev)
    print("TEST:",tm)

    # ── plot learning curves
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(tr_loss_hist,label='Train'); plt.plot(va_loss_hist,label='Val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(va_f1_hist,label='Val F1'); plt.title('Validation F1')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('learning_curves.png',dpi=300)

if __name__=='__main__':
    main()
