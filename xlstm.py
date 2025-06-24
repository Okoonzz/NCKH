import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# xLSTM imports from NX-AI library
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig
)

def load_json(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)

class SequenceDataset(Dataset):
    """
    Dataset combining all features (as in graphml) into a token sequence for xLSTM.
    Order: first 1000 API calls, then behavior_summary, dropped_files,
    signatures, processes, network.
    Expects root_dir with 'benign' and 'ransomware' subfolders containing JSON files.
    """
    def __init__(self, root_dir, max_seq_len=1500):
        self.samples = []
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        self.max_seq_len = max_seq_len
        for label_name, label in [('json-atb-benign-507', 0), ('ransom-5xx-new/ransomware', 1)]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith('.json'):
                    continue
                path = os.path.join(folder, fname)
                feat = load_json(path)
                tokens = []
                # First: API call tokens (first 1000)
                for call in feat.get('api_call_sequence', [])[:1000]:
                    tokens.append(f"api:{call.get('api','')}")
                # behavior_summary
                for ft, vals in feat.get('behavior_summary', {}).items():
                    for v in vals:
                        tokens.append(f"feature:{ft}:{v}")
                # dropped_files
                for d in feat.get('dropped_files', []):
                    if isinstance(d, dict):
                        tokens.append(f"dropped_file:{d.get('filepath','')}")
                    else:
                        tokens.append(f"dropped_file:{d}")
                # signatures
                for sig in feat.get('signatures', []):
                    tokens.append(f"signature:{sig.get('name','')}")
                # processes
                for p in feat.get('processes', []):
                    tokens.append(f"process:{p.get('name','')}")
                # network
                for proto, entries in feat.get('network', {}).items():
                    for entry in entries:
                        if isinstance(entry, dict):
                            dst = entry.get('dst') or entry.get('dst_ip','')
                            port = entry.get('dst_port') or entry.get('port','')
                            tokens.append(f"network:{proto}:{dst}:{port}")
                        else:
                            tokens.append(f"network:{proto}:{entry}")
                # Update vocab and samples
                for tok in tokens:
                    if tok not in self.vocab:
                        self.vocab[tok] = idx
                        idx += 1
                self.samples.append((tokens, label))
        print(f"Loaded {len(self.samples)} samples; Vocab size = {len(self.vocab)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        tokens, label = self.samples[i]
        idxs = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        if len(idxs) >= self.max_seq_len:
            idxs = idxs[:self.max_seq_len]
        else:
            idxs = idxs + [self.vocab['<PAD>']] * (self.max_seq_len - len(idxs))
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class XLSTMClassifier(nn.Module):
    """
    Wrapper using NX-AI xLSTMBlockStack for feature token sequences.
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        max_seq_len=1500,
        num_blocks=1,
        dropout=0.3
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.config = xLSTMBlockStackConfig(
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
            context_length=max_seq_len,
            num_blocks=num_blocks,
            embedding_dim=embed_dim,
            slstm_at=[1]
        )
        self.xlstm = xLSTMBlockStack(self.config)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, seq):
        emb = self.embed(seq)
        out = self.xlstm(emb)
        seq_emb = out.mean(dim=1)
        return self.fc(seq_emb).squeeze(1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for seq, labels in loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            logits = model(seq)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def main():
    data_dir = "/kaggle/input"
    max_seq_len = 1500
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 20
    embed_dim = 128
    num_blocks = 3
    dropout = 0.3
    train_losses, test_losses, test_accs  = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = SequenceDataset(data_dir, max_seq_len=max_seq_len)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"Train samples: {train_size}, Test samples: {test_size}")

    model = XLSTMClassifier(
        vocab_size=len(dataset.vocab),
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        dropout=dropout
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)

        train_losses.append(tr_loss)
        test_losses.append(val_loss)
        test_accs.append(val_acc)
        
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_xlstm_model.pth')
            print(f"  → Saved best model (Val Acc: {best_acc:.4f})")
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses)+1),  test_losses,  label='Test Loss')
    plt.plot(range(1, len(test_accs)+1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Lưu ra file trước khi hiển thị
    plt.savefig('loss_convergence.png', dpi=300) 
if __name__ == '__main__':
    main()
