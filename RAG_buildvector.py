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