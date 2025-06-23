import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# === 1. CHỌN DEVICE ===
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 2. ĐỊNH NGHĨA MÔ HÌNH GCN (2 LỚP) ===
class GCN2Layer(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout_p: float = 0.7):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1   = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2   = BatchNorm(hidden_channels)
        self.dropout_p = dropout_p
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# === 3. CUSTOM DATASET: Kiểm tra NaN/Inf + Scale numeric ===
class GraphFileDataset(Dataset):
    def __init__(self, root_dir: str, subdirs=('benign','ransomware')):
        """
        root_dir: folder chứa các folder con benign/ ransomware/
        subdirs: danh sách folder con
        """
        self.file_list = []
        for sub in subdirs:
            folder = os.path.join(root_dir, sub)
            if not os.path.isdir(folder):
                continue
            lab = 0 if sub == 'benign' else 1
            for fname in os.listdir(folder):
                if not fname.endswith(".pt"):
                    continue
                self.file_list.append((os.path.join(folder, fname), lab))

        # Bây giờ chỉ có 1 chiều numeric (size hoặc severity)
        self.max_numeric_features = 1
        self.scale_factor = 1e6  # chia cho 1e6 để scale xuống

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, lab = self.file_list[idx]
        data = torch.load(path, weights_only=False)

        # 1) Kiểm tra NaN/Inf trong x
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) Scale numeric feature cuối cùng (1 chiều)
        in_ch = data.x.size(1)
        numeric_start = in_ch - self.max_numeric_features
        numeric_end   = in_ch
        data.x[:, numeric_start:numeric_end] /= self.scale_factor

        # Gán lại label chắc chắn đúng
        data.y = torch.tensor([lab], dtype=torch.long)
        return data

# === 4. HÀM TRAIN/TỨC TEST MỘT EPOCH (KHÔNG DÙNG AMP) ===
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)

        del batch
        torch.cuda.empty_cache()

    return total_loss / total, correct / total

@torch.no_grad()
def test_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)

        del batch
        torch.cuda.empty_cache()

    return total_loss / total, correct / total

# === 5. MAIN: CHỦ ĐỘNG CHẠY TRAIN/TEST ===
if __name__ == "__main__":
    root_pt = "pyg_data_DistilBERT"  # thư mục chứa "benign/" và "ransomware/"

    # 5.1 Tạo dataset
    full_dataset = GraphFileDataset(root_pt, subdirs=('benign','ransomware'))
    print(f"Total graphs available: {len(full_dataset)}")

    # 5.2 Split train/test (80:20)
    total = len(full_dataset)
    n_train = int(0.8 * total)
    n_test  = total - n_train
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_test])
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

    # 5.3 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 5.4 Lấy in_channels từ 1 sample
    sample_path, _ = full_dataset.file_list[0]
    sample_data   = torch.load(sample_path, weights_only=False)
    in_channels   = sample_data.x.size(1)  # bây giờ = 775
    print("Feature dimension (in_channels):", in_channels)

    # 5.5 Khởi tạo model, optimizer, loss, scheduler
    hidden_channels = 64
    num_classes     = 2
    dropout_p       = 0.6

    model = GCN2Layer(in_channels, hidden_channels, num_classes, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5.6 Vòng lặp huấn luyện
    best_test_acc = 0.0
    num_epochs = 50
    train_losses, test_losses, test_accs  = [], [], []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss,  test_acc  = test_epoch(model, test_loader,  criterion)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  → Saved best model (Test Acc: {best_test_acc:.4f})\n")

    print("Training completed. Best Test Acc:", best_test_acc)



    # patience = 6            # nếu test loss không giảm sau 6 epoch thì dừng
    # best_loss = float('inf')
    # epochs_no_improve = 0

    # # --- lists để lưu loss mỗi epoch ---
    # train_losses, test_losses, test_accs  = [], [], []
    # num_epochs = 30

    # for epoch in range(1, num_epochs + 1):
    #     tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    #     te_loss, te_acc = test_epoch(model, test_loader, criterion)
    #     scheduler.step(te_loss)

    #     train_losses.append(tr_loss)
    #     test_losses.append(te_loss)
    #     test_accs.append(te_acc)

    #     print(f"Epoch {epoch:02d} | "
    #           f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
    #           f"Test  Loss: {te_loss:.4f}, Test  Acc: {te_acc:.4f}")

    #     # --- kiểm tra early stopping ---
    #     if te_loss < best_loss:
    #         best_loss = te_loss
    #         epochs_no_improve = 0
    #         torch.save(model.state_dict(), "best_model.pth")
    #         print(f"  → Test loss giảm, lưu model (Loss: {best_loss:.4f})\n")
    #     else:
    #         epochs_no_improve += 1
    #         print(f"  → Không cải thiện test loss: {epochs_no_improve}/{patience}\n")
    #         if epochs_no_improve >= patience:
    #             print("Early stopping triggered. Dừng huấn luyện.\n")
    #             break

    # --- VẼ đồ thị hội tụ loss ---
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
    plt.show()
