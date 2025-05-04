import os
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset

# === Chuyển file .graphml thành PyG Data ===
def graphml_to_data(path):
    G = nx.read_graphml(path)

    node_list = list(G.nodes)
    node_idx_map = {nid: i for i, nid in enumerate(node_list)}

    # Edge index (undirected)
    edge_index = []
    for src, dst in G.edges():
        i = node_idx_map[src]
        j = node_idx_map[dst]
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # One-hot node type
    type_to_idx = {"api": 0, "feature": 1, "tactic": 2}
    x = []
    for nid in node_list:
        node_type = G.nodes[nid].get("node_type", "")
        onehot = [0] * len(type_to_idx)
        if node_type in type_to_idx:
            onehot[type_to_idx[node_type]] = 1
        x.append(onehot)
    x = torch.tensor(x, dtype=torch.float)

    # Graph-level label
    label = int(G.graph.get("label", 0))
    y = torch.tensor([label], dtype=torch.float)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.graph_id = G.graph.get("graph_id", os.path.basename(path))
    return data

# === Dataset loader ===
class GraphMLDataset(InMemoryDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.graph_paths = []

        for subdir in ["benign", "ransomware"]:
            full_path = os.path.join(root_dir, subdir)
            for fname in os.listdir(full_path):
                if fname.endswith(".graphml"):
                    self.graph_paths.append(os.path.join(full_path, fname))

        self.data_list = [graphml_to_data(p) for p in self.graph_paths]

        # Save to .pt
        save_path = os.path.join(root_dir, "dataset_pyg.pt")
        torch.save(self.data_list, save_path)
        print(f"✅ Saved dataset with {len(self.data_list)} graphs → {save_path}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
