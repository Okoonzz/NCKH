## dung co huong va node 2 loai ##
# import json
# import networkx as nx

# # === 1) Load feature2.json ===
# with open("features2.json", "r", encoding="utf-8") as f:
#     features = json.load(f)

# api_sequence = features.get("api_call_sequence", [])

# # === 2) Parse contexts from feature_context_all_results11.txt ===
# contexts = {}
# with open("feature_context_all_results11.txt", "r", encoding="utf-8") as f:
#     lines = [l.rstrip("\n") for l in f]

# i = 0
# while i < len(lines):
#     line = lines[i]
#     if line.startswith("Feature: "):
#         # a) Extract feature key
#         key = line[len("Feature: "):]
#         feat_type, feat_val = map(str.strip, key.split("=", 1))
#         # initialize context dict
#         ctx = {"context": "", "sources": "", "tactics": ""}

#         # b) Advance into the block
#         i += 1
#         while i < len(lines) and not lines[i].startswith("Feature: "):
#             l = lines[i]

#             # capture multi-line Context
#             if l.startswith("→ Context:"):
#                 i += 1  # skip the marker line
#                 ctxt_lines = []
#                 # gather all following lines that are NOT new markers
#                 while i < len(lines) and not lines[i].startswith("→ "):
#                     text = lines[i].strip()
#                     if text:
#                         ctxt_lines.append(text)
#                     i += 1
#                 ctx["context"] = " ".join(ctxt_lines)
#                 continue

#             # capture Sources
#             if l.startswith("→ Sources:"):
#                 ctx["sources"] = l[len("→ Sources:"):].strip()

#             # capture Tactic(s)
#             if l.startswith("→ Tactic(s):"):
#                 ctx["tactics"] = l[len("→ Tactic(s):"):].strip()

#             i += 1

#         # save into contexts map
#         print(ctx)
#         contexts[(feat_type, feat_val)] = ctx
#     else:
#         i += 1

# # optional test print
# # for k,v in list(contexts.items())[:5]:
# #     print(k, "→", v)

# # === 3) Flatten features into list of (type, value) ===
# def flatten_features(obj, prefix=None):
#     if prefix is None:
#         prefix = []
#     items = []
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             items.extend(flatten_features(v, prefix + [k]))
#     elif isinstance(obj, list):
#         feat_type = "_".join(prefix)
#         for val in obj:
#             val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
#             items.append((feat_type, val_str))
#     else:
#         feat_type = "_".join(prefix)
#         items.append((feat_type, str(obj)))
#     return items

# flat_feats = flatten_features(features)

# # === 4) Build the directed graph === co huong
# G = nx.DiGraph()

# # 4a) Add feature nodes with context metadata
# for feat_type, feat_val in flat_feats:
#     node_id = f"feature::{feat_type}::{feat_val}"
#     attrs = {
#         "node_type": "feature",
#         "feat_type": feat_type,
#         "feat_val": feat_val
#     }
#     meta = contexts.get((feat_type, feat_val), {})
#     attrs.update(meta)
#     G.add_node(node_id, **attrs)

# # 4b) Add API-call nodes with context metadata
# for idx, api in enumerate(api_sequence):
#     node_id = f"api::{idx}::{api['api']}"
#     attrs = {
#         "node_type": "api",
#         "api": api.get("api", ""),
#         "category": api.get("category", ""),
#         "arguments": api.get("arguments", {})
#     }
#     api_val_str = json.dumps(api, ensure_ascii=False)
#     meta = contexts.get(("api_call_sequence", api_val_str), {})
#     attrs.update(meta)
#     G.add_node(node_id, **attrs)

# # 4c) Link API calls in sequence
# for i in range(len(api_sequence) - 1):
#     src = f"api::{i}::{api_sequence[i]['api']}"
#     dst = f"api::{i+1}::{api_sequence[i+1]['api']}"
#     G.add_edge(src, dst, relation="sequence")

# # 4d) Link features → API if feature value appears in any argument
# for feat_type, feat_val in flat_feats:
#     fnode = f"feature::{feat_type}::{feat_val}"
#     for idx, api in enumerate(api_sequence):
#         api_args = api.get("arguments", {})
#         if any(str(feat_val) in str(v) for v in api_args.values()):
#             anode = f"api::{idx}::{api['api']}"
#             G.add_edge(fnode, anode, relation="related")

# # === 5) Sanitize attributes: convert dict/list → JSON string ===
# import json as _json
# for nid, data in G.nodes(data=True):
#     for k, v in list(data.items()):
#         if isinstance(v, (dict, list)):
#             data[k] = _json.dumps(v, ensure_ascii=False)

# for u, v, data in G.edges(data=True):
#     for k, val in list(data.items()):
#         if isinstance(val, (dict, list)):
#             data[k] = _json.dumps(val, ensure_ascii=False)

# # === 6) Save to GraphML ===
# nx.write_graphml(G, "feature_api_graph2.graphml")
# print("✅ Graph saved to feature_api_graph2.graphml")

## dung co huong va node 2 loai ##
###############################################################################################################
## dung vo huong va node 5 loai ##

import json
import networkx as nx
from itertools import combinations

# === 1) Load feature2.json ===
with open("features2.json", "r", encoding="utf-8") as f:
    features = json.load(f)

api_sequence = features.get("api_call_sequence", [])

# === 2) Parse contexts from feature_context_all_results11.txt ===
contexts = {}
with open("feature_context_all_results11.txt", "r", encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("Feature: "):
        key = line[len("Feature: "):]
        feat_type, feat_val = map(str.strip, key.split("=", 1))
        ctx = {"context": "", "sources": "", "tactics": ""}

        i += 1
        while i < len(lines) and not lines[i].startswith("Feature: "):
            l = lines[i]
            if l.startswith("→ Context:"):
                i += 1
                ctxt_lines = []
                while i < len(lines) and not lines[i].startswith("→ "):
                    text = lines[i].strip()
                    if text:
                        ctxt_lines.append(text)
                    i += 1
                ctx["context"] = " ".join(ctxt_lines)
                continue
            if l.startswith("→ Sources:"):
                ctx["sources"] = l[len("→ Sources:"):].strip()
            if l.startswith("→ Tactic(s):"):
                ctx["tactics"] = l[len("→ Tactic(s):"):].strip()
            i += 1

        contexts[(feat_type, feat_val)] = ctx
    else:
        i += 1

# === 3) Flatten features ===
def flatten_features(obj, prefix=None):
    if prefix is None:
        prefix = []
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.extend(flatten_features(v, prefix + [k]))
    elif isinstance(obj, list):
        feat_type = "_".join(prefix)
        for val in obj:
            val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
            items.append((feat_type, val_str))
    else:
        feat_type = "_".join(prefix)
        items.append((feat_type, str(obj)))
    return items

flat_feats = flatten_features(features)

# === 4) Build undirected graph for GCN ===
G = nx.Graph()

# 4a) Add feature nodes
for feat_type, feat_val in flat_feats:
    node_id = f"feature::{feat_type}::{feat_val}"
    attrs = {
        "node_type": "feature",
        "feat_type": feat_type,
        "feat_val": feat_val
    }
    meta = contexts.get((feat_type, feat_val), {})
    attrs.update(meta)
    G.add_node(node_id, **attrs)

# 4b) Add API nodes
for idx, api in enumerate(api_sequence):
    node_id = f"api::{idx}::{api['api']}"
    attrs = {
        "node_type": "api",
        "api": api.get("api", ""),
        "category": api.get("category", ""),
        "arguments": api.get("arguments", {})
    }
    api_val_str = json.dumps(api, ensure_ascii=False)
    meta = contexts.get(("api_call_sequence", api_val_str), {})
    attrs.update(meta)
    G.add_node(node_id, **attrs)

# 4c) API → API sequence
for i in range(len(api_sequence) - 1):
    src = f"api::{i}::{api_sequence[i]['api']}"
    dst = f"api::{i+1}::{api_sequence[i+1]['api']}"
    G.add_edge(src, dst, relation="sequence")

# 4d) Feature ↔ API nếu value xuất hiện trong arg
for feat_type, feat_val in flat_feats:
    fnode = f"feature::{feat_type}::{feat_val}"
    for idx, api in enumerate(api_sequence):
        api_args = api.get("arguments", {})
        if any(str(feat_val) in str(v) for v in api_args.values()):
            anode = f"api::{idx}::{api['api']}"
            G.add_edge(fnode, anode, relation="arg_related")

# 4e) Feature ↔ Feature nếu cùng tactic
feature_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "feature"]
for n1, n2 in combinations(feature_nodes, 2):
    t1, t2 = G.nodes[n1].get("tactics"), G.nodes[n2].get("tactics")
    if t1 and t2 and t1 == t2 and t1 != "":
        G.add_edge(n1, n2, relation="same_tactic")

# 4f) Feature ↔ API nếu cùng tactic
api_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "api"]
for f in feature_nodes:
    ftac = G.nodes[f].get("tactics", "")
    for a in api_nodes:
        atac = G.nodes[a].get("tactics", "")
        if ftac and atac and ftac == atac and ftac != "":
            G.add_edge(f, a, relation="tactic_linked")

# 4g) Thêm tactic node và liên kết tactic
tactic_set = set()
for n, d in list(G.nodes(data=True)):
    tacs = d.get("tactics", "")
    for t in [t.strip() for t in tacs.split(",") if t.strip()]:
        tid = f"tactic::{t}"
        tactic_set.add(tid)
        if not G.has_node(tid):
            G.add_node(tid, node_type="tactic", tactic=t)
        G.add_edge(n, tid, relation="tactic_of")

# 5) Sanitize attributes to string
for nid, data in G.nodes(data=True):
    for k, v in list(data.items()):
        if isinstance(v, (dict, list)):
            data[k] = json.dumps(v, ensure_ascii=False)

for u, v, data in G.edges(data=True):
    for k, val in list(data.items()):
        if isinstance(val, (dict, list)):
            data[k] = json.dumps(val, ensure_ascii=False)

# 6) Save
nx.write_graphml(G, "feature_api_graph3.graphml")
print("✅ GCN Graph saved to feature_api_graph3.graphml")
