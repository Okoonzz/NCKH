import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from data.sequence_vector import load_report, build_text_event_sequence
from data.vocab import build_vocab, encode_sequence
from model.xlstm_classifier import xLSTMClassifier

# Cấu hình đường dẫn và nhãn
report_paths = ['reports/report.json']  # hoặc list nhiều file
labels = [1]  # 1=ransomware,0=benign tương ứng

# Load report và build chuỗi sự kiện
reports = [load_report(p) for p in report_paths]
sequences = [build_text_event_sequence(r) for r in reports]

# Build và lưu vocab
vocab = build_vocab(sequences)
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False)

# Encode sequences và tạo dataset
max_len = 50
X = [encode_sequence(seq, vocab, max_len) for seq in sequences]
y = labels

dataset = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float).unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Khởi tạo model, optimizer, loss
model = xLSTMClassifier(vocab_size=len(vocab), embed_dim=128, context_length=max_len)
opt = Adam(model.parameters(), lr=1e-3)
loss_fn = BCEWithLogitsLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    for xb, yb in dataloader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Lưu checkpoint
torch.save(model.state_dict(), 'model.pt')