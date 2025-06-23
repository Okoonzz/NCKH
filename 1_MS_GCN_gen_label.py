import os
import networkx as nx

# === Đường dẫn đến thư mục dataset
root_dir = "graph_1000_api"
# root_dir = "graph_MS_contexts"

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


# G2 = nx.read_graphml("graph_MS_contexts/benign/report_0bf9cfc88d56b52982655b56d577f69d2ee4b136908684015001c52c64f0a84b.exe.graphml")
# print(G2.graph)