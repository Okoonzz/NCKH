import os
import networkx as nx

# === Đường dẫn đến thư mục dataset
root_dir = "dataset"

# === Duyệt cả benign và ransomware
label_map = {
    "benign": 0,
    "ransomware": 1
}

for subdir in label_map:
    label = label_map[subdir]
    folder = os.path.join(root_dir, subdir)
    for fname in os.listdir(folder):
        if fname.endswith(".graphml"):
            path = os.path.join(folder, fname)
            print(f"📄 Gán label={label} cho {fname}")
            G = nx.read_graphml(path)
            G.graph["label"] = label
            G.graph["graph_id"] = fname  # để trace sau này
            nx.write_graphml(G, path)
