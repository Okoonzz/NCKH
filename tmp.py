# import os
# import json
# from collections import defaultdict

# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import Chroma

# # Paths to your local MITRE CTI JSON directories
# MALWARE_DIR         = "cti/enterprise-attack/malware"
# RELATIONSHIP_DIR    = "cti/enterprise-attack/relationship"
# ATTACK_PATTERN_DIR  = "cti/enterprise-attack/attack-pattern"
# PERSIST_DIR         = "vectordb_mitre_attack"
# PREVIEW_FILE        = "mitre_attack_chunks_preview.txt"

# # 1) Load all techniques
# tech_map = {}
# for fn in os.listdir(ATTACK_PATTERN_DIR):
#     with open(os.path.join(ATTACK_PATTERN_DIR, fn), encoding="utf-8") as f:
#         bundle = json.load(f)
#     for obj in bundle.get("objects", []):
#         if obj.get("type") == "attack-pattern":
#             ext_id = next(
#                 (r["external_id"] for r in obj.get("external_references", [])
#                  if r.get("source_name") == "mitre-attack"),
#                 "N/A"
#             )
#             tech_map[obj["id"]] = {
#                 "external_id": ext_id,
#                 "name": obj.get("name", "Unknown"),
#                 "description": obj.get("description", "").replace("\n", " ")
#             }

# # 2) Load all malware names (for example contexts)
# malware_map = {}
# for fn in os.listdir(MALWARE_DIR):
#     with open(os.path.join(MALWARE_DIR, fn), encoding="utf-8") as f:
#         bundle = json.load(f)
#     for obj in bundle.get("objects", []):
#         if obj.get("type") == "malware":
#             malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# # 3) Gather "uses" examples per technique
# examples = defaultdict(list)
# for fn in os.listdir(RELATIONSHIP_DIR):
#     with open(os.path.join(RELATIONSHIP_DIR, fn), encoding="utf-8") as f:
#         bundle = json.load(f)
#     for rel in bundle.get("objects", []):
#         if (rel.get("type") == "relationship" and
#             rel.get("relationship_type") == "uses"):
#             mid = rel.get("source_ref")
#             tid = rel.get("target_ref")
#             desc = rel.get("description", "").strip()
#             if mid in malware_map and tid in tech_map and desc:
#                 examples[tid].append(f"{malware_map[mid]}: {desc}")

# # 4) Build one chunk per technique, including description + examples
# chunks = []
# metadatas = []

# for tid, info in tech_map.items():
#     lines = [
#         f"{info['external_id']} – {info['name']}",
#         f"Description: {info['description']}"
#     ]
#     if examples[tid]:
#         lines.append("Examples:")
#         for ex in examples[tid]:
#             lines.append(f"- {ex}")
#     text = "\n".join(lines)
#     chunks.append(text)
#     metadatas.append({
#         "external_id": info["external_id"],
#         "name": info["name"]
#     })

# # # 4.5) **Preview**: ghi tất cả chunks ra file để bạn kiểm tra
# # with open(PREVIEW_FILE, "w", encoding="utf-8") as out:
# #     for chunk in chunks:
# #         out.write(chunk)
# #         out.write("\n\n---\n\n")
# # print(f"📄 Đã ghi preview chunks vào: {PREVIEW_FILE}")

# # 5) Index into Chroma
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# db = Chroma.from_texts(
#     texts=chunks,               # your list of technique+example texts
#     embedding=embedding,        # correct param name is `embedding`
#     metadatas=metadatas,
#     persist_directory=PERSIST_DIR
# )
# db.persist()

# print(f"✅ Built MITRE ATT&CK vector DB with {len(chunks)} techniques in '{PERSIST_DIR}'.")
#####################################################################################################################################

# import json

# def extract_list(d, key):
#     return d.get(key, []) if isinstance(d.get(key), list) else []

# def main():
#     with open("report.json", "r", encoding="utf-8") as f:
#         report = json.load(f)

#     features = {}

#     # 1) Behavior summary
#     summary = report.get("behavior", {}).get("summary", {})
#     behavior_keys = [
#         "file_created", "file_deleted", "file_read", "file_recreated", "file_exists",
#         "file_opened", "regkey_opened", "regkey_read", "regkey_written",
#         "dll_loaded", "executed_commands", "directory_created",
#         "mutex", "processes_created", "resolves_host"
#     ]
#     features["behavior_summary"] = {k: extract_list(summary, k) for k in behavior_keys}

#     # 2) Dropped files
#     features["dropped_files"] = report.get("dropped", [])

#     # 3) API statistics
#     features["api_statistics"] = report.get("behavior", {}).get("apistats", {})

#     # 4) API call sequence (ordered)
#     api_calls = []
#     for proc in report.get("behavior", {}).get("processes", []):
#         for call in proc.get("calls", []):
#             api_calls.append({
#                 "time": call.get("time"),
#                 "api": call.get("api"),
#                 "pid": proc.get("pid"),
#                 "arguments": call.get("arguments", {}),
#                 "category": call.get("category", "")
#             })
#     # Sort theo thời gian
#     api_calls.sort(key=lambda x: x["time"])
#     features["api_call_sequence"] = api_calls

#     # 5) Network activity
#     net = report.get("network", {})
#     features["network"] = {
#         "http": net.get("http", []),
#         "tcp": net.get("tcp", []),
#         "udp": net.get("udp", []),
#         "dns": net.get("dns", []),
#         "hosts": net.get("hosts", []),
#         "icmp": net.get("icmp", []),
#         "smtp": net.get("smtp", [])
#     }

#     # 6) Static imports
#     static = report.get("static", {})
#     imports = []
#     for dll in static.get("pe_imports", []):
#         for imp in dll.get("imports", []):
#             name = imp.get("name")
#             if name:
#                 imports.append(name)
#     features["static_imports"] = imports

#     # 7) Signatures - TTP only
#     ttps = {}
#     for sig in report.get("signatures", []):
#         for ttp_id, ttp in sig.get("ttp", {}).items():
#             ttps[ttp_id] = ttp.get("short", "")
#     features["signatures_ttp"] = ttps

#     # 8) Signatures - Full
#     sig_full = []
#     for sig in report.get("signatures", []):
#         sig_full.append({
#             "name": sig.get("name"),
#             "description": sig.get("description"),
#             "severity": sig.get("severity"),
#             "ttp": sig.get("ttp", {}),
#             "markcount": sig.get("markcount", 0),
#             "marks": sig.get("marks", [])
#         })
#     features["signatures_full"] = sig_full

#     # 9) Process details
#     processes = []
#     for p in report.get("behavior", {}).get("processes", []):
#         processes.append({
#             "name": p.get("process_name"),
#             "path": p.get("process_path"),
#             "pid": p.get("pid"),
#             "cmdline": p.get("command_line"),
#             "modules": [
#                 m.get("basename")
#                 for m in p.get("modules", [])
#                 if m.get("basename")
#             ]
#         })
#     features["processes"] = processes

#     # 10) Save output
#     with open("features2.json", "w", encoding="utf-8") as f:
#         json.dump(features, f, indent=2, ensure_ascii=False)

#     print("✅ Đã trích xuất đầy đủ features vào file features2.json")

# if __name__ == "__main__":
#     main()

#####################################################################################################################################################################################################################################################################




import os
import json
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === Cấu hình ===
BASE_DIR = "cti/enterprise-attack"
ATTACK_PATTERN_DIR = os.path.join(BASE_DIR, "attack-pattern")
MALWARE_DIR = os.path.join(BASE_DIR, "malware")
RELATIONSHIP_DIR = os.path.join(BASE_DIR, "relationship")
TACTIC_DIR = os.path.join(BASE_DIR, "x-mitre-tactic")
PERSIST_DIR = "vectordb_mitre_attack_tatic2"  # vector DB đầu ra
PREVIEW_FILE = "mitre_attack_chunks_preview_tatic2.txt"

# === 1. Load mapping: phase_name → tactic (external_id + name) ===
phase_to_tactic = {}
for fn in os.listdir(TACTIC_DIR):
    with open(os.path.join(TACTIC_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-tactic":
            phase_name = obj.get("x_mitre_shortname")  # e.g., "credential-access"
            ext_id = next((r["external_id"] for r in obj.get("external_references", [])
                           if r.get("source_name") == "mitre-attack"), None)
            name = obj.get("name")
            if phase_name and ext_id and name:
                phase_to_tactic[phase_name] = {
                    "external_id": ext_id,
                    "name": name
                }

# === 2. Load all attack-patterns ===
tech_map = {}
for fn in os.listdir(ATTACK_PATTERN_DIR):
    with open(os.path.join(ATTACK_PATTERN_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        if obj.get("type") == "attack-pattern":
            ext_id = next((r["external_id"] for r in obj.get("external_references", [])
                           if r.get("source_name") == "mitre-attack"), "N/A")
            tactic_objs = obj.get("kill_chain_phases", [])
            tactics = []
            for t in tactic_objs:
                phase = t.get("phase_name")
                if phase in phase_to_tactic:
                    tactic_info = phase_to_tactic[phase]
                    tactics.append(f"{tactic_info['external_id']} – {tactic_info['name']}")
            tech_map[obj["id"]] = {
                "external_id": ext_id,
                "name": obj.get("name", "Unknown"),
                "description": obj.get("description", "").replace("\n", " "),
                "tactics": sorted(set(tactics))
            }

# === 3. Load malware (for examples) ===
malware_map = {}
for fn in os.listdir(MALWARE_DIR):
    with open(os.path.join(MALWARE_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        if obj.get("type") == "malware":
            malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# === 4. Collect malware → technique examples ===
examples = defaultdict(list)
for fn in os.listdir(RELATIONSHIP_DIR):
    with open(os.path.join(RELATIONSHIP_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for rel in bundle.get("objects", []):
        if rel.get("type") == "relationship" and rel.get("relationship_type") == "uses":
            mid = rel.get("source_ref")
            tid = rel.get("target_ref")
            desc = rel.get("description", "").strip()
            if mid in malware_map and tid in tech_map and desc:
                examples[tid].append(f"{malware_map[mid]}: {desc}")

# === 5. Build vector DB chunks ===
chunks = []
metadatas = []

for tid, info in tech_map.items():
    lines = [
        f"{info['external_id']} – {info['name']}",
        f"Description: {info['description']}"
    ]
    if examples[tid]:
        lines.append("Examples:")
        for ex in examples[tid]:
            lines.append(f"- {ex}")
    text = "\n".join(lines)
    chunks.append(text)
    metadatas.append({
        "external_id": info["external_id"],
        "name": info["name"],
        "tactic": "; ".join(info["tactics"])  # ✔ tactic nằm ở metadata
    })

# === 6. Save preview (tùy chọn) ===
with open(PREVIEW_FILE, "w", encoding="utf-8") as out:
    for chunk in chunks:
        out.write(chunk + "\n\n---\n\n")

# === 7. Build vector DB ===
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(PERSIST_DIR):
    import shutil
    shutil.rmtree(PERSIST_DIR)

db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding,
    metadatas=metadatas,
    persist_directory=PERSIST_DIR
)
db.persist()

print(f"✅ Đã build vector DB chuẩn với {len(chunks)} kỹ thuật tại '{PERSIST_DIR}'")
