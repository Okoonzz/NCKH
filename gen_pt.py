import os
import glob
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------- Interactive selection for embedding model -------
print("Choose embedding model:")
print("1: DistilBERT (distilbert-base-uncased)")

EMBED_MODEL = 'distilbert-base-uncased'
OUTPUT_DIR = 'pyg_data_DistilBERT'
print(f"Selected model: {EMBED_MODEL}")
print(f"Output directory: {OUTPUT_DIR}")
# ---------------------------------------------------------

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 100
NUM_WORKERS = None
SEP = '<SEP>'  # safe separator for text fields

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained(EMBED_MODEL)
embed_model = DistilBertModel.from_pretrained(EMBED_MODEL).to(device)
model_lock = Lock()

# Encode text batch
def encode_text_batch(texts):
    embeddings = []
    texts = [' ' if not t or t == '{}' else t for t in texts]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True
        ).to(device)
        with model_lock:
            with torch.no_grad():
                outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
        embeddings.extend(emb)
        del inputs, outputs, emb
        torch.cuda.empty_cache()
    return embeddings

# Lấy đặc trưng cho từng node 
def graphml_to_pyg_data(path):
    try:
        G = nx.read_graphml(path, node_type=str)
        mapping = {n: i for i, n in enumerate(G.nodes())}
        texts = []
        for n, data in G.nodes(data=True):
            nt = data.get('node_type', '')
            parts = [nt]
            if nt == 'feature':
                parts.append(f"feature_type:{data.get('feature_type','')}")
                parts.append(f"value:{data.get('feature_value','')}")
            elif nt == 'api':
                parts.append(f"api:{data.get('api','')}")
                parts.append(f"args:{data.get('arguments','{}')}")
            elif nt == 'process':
                parts.extend([
                    f"name:{data.get('name','')}",
                    f"path:{data.get('path','')}",
                    f"cmdline:{data.get('cmdline','')}"
                ])
            elif nt == 'dropped_file':
                parts.extend([
                    f"filepath:{data.get('filepath','')}",
                    f"size:{data.get('size',0)}"
                ])
            elif nt == 'network':
                parts.append(f"category:{data.get('category','')}")
                parts.append(json.dumps(json.loads(data.get('details','{}')), ensure_ascii=False))
            elif nt == 'signature':
                parts.extend([
                    f"name:{data.get('name','')}",
                    f"description:{data.get('description','')}",
                    f"severity:{data.get('severity',0)}"
                ])
            texts.append(SEP.join(parts))
        embs = encode_text_batch(texts)
        x = torch.tensor(embs, dtype=torch.float)
        print(f"x shape: {x.shape}")
        edge_idx = torch.tensor(
            [[mapping[u], mapping[v]] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()
        return Data(x=x, edge_index=edge_idx, graph_id=G.graph.get('graph_id', os.path.basename(path)))
    except Exception as e:
        print(f"Error {path}: {e}")
        return None

# Worker to save
def process_and_save(path, out_dir):
    data = graphml_to_pyg_data(path)
    if data is None:
        return
    data.x = data.x.cpu(); data.edge_index = data.edge_index.cpu()
    name = os.path.splitext(os.path.basename(path))[0]
    torch.save(data, os.path.join(out_dir, f"{name}.pt"))
    print(f"Saved {name}.pt")

# Multi-threaded batch convert
def batch_convert(graphml_dir, subdirs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in subdirs:
        in_d = os.path.join(graphml_dir, sub)
        out_d = os.path.join(OUTPUT_DIR, sub)
        os.makedirs(out_d, exist_ok=True)
        paths = glob.glob(os.path.join(in_d, '*.graphml'))
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            for p in paths:
                ex.submit(process_and_save, p, out_d)

if __name__ == '__main__':
    batch_convert(
        os.getenv('GRAPHML_DIR', 'graph_1000_api'),
        ['ransomware', 'benign_nnnnnnnnnnnnnnnn']
    )
