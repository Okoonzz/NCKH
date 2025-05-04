
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from GCN_load_all_graphml_Input import GraphMLDataset
from GCN_model import GCN
from sklearn.metrics import accuracy_score, f1_score

# 1) Load dataset (chỉ 1 graph)
dataset = GraphMLDataset("dataset/")
assert len(dataset) == 1, "Hiện chỉ có đúng 1 graph trong dataset"
data = dataset[0]

# 2) DataLoader cho 1 mẫu, batch_size=1
loader = DataLoader([data], batch_size=1, shuffle=False)

# 3) Khởi tạo model + optimizer + loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = data.x.size(1)
model = GCN(in_channels=in_dim, hidden_channels=64, num_layers=2, dropout=0.5).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
crit  = torch.nn.BCEWithLogitsLoss()

# 4) Train & Test trên cùng 1 graph
for epoch in range(1, 11):
    model.train()
    total_loss = 0.0

    # train step
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = crit(logits, batch.y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    # eval step (trên cùng graph)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits).cpu()
            pred = (probs > 0.5).float()
            ys.append(batch.y.cpu().item())
            ps.append(pred.item())

    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, zero_division=0)

    print(f"Epoch {epoch:02d} | loss: {total_loss:.4f} | acc: {acc:.4f} | f1: {f1:.4f}")
