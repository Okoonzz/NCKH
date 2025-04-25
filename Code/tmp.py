import os
import json
from collections import defaultdict

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Paths to your local MITRE CTI JSON directories
MALWARE_DIR         = "cti/enterprise-attack/malware"
RELATIONSHIP_DIR    = "cti/enterprise-attack/relationship"
ATTACK_PATTERN_DIR  = "cti/enterprise-attack/attack-pattern"
PERSIST_DIR         = "vectordb_mitre_attack"
PREVIEW_FILE        = "mitre_attack_chunks_preview.txt"

# 1) Load all techniques
tech_map = {}
for fn in os.listdir(ATTACK_PATTERN_DIR):
    with open(os.path.join(ATTACK_PATTERN_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        if obj.get("type") == "attack-pattern":
            ext_id = next(
                (r["external_id"] for r in obj.get("external_references", [])
                 if r.get("source_name") == "mitre-attack"),
                "N/A"
            )
            tech_map[obj["id"]] = {
                "external_id": ext_id,
                "name": obj.get("name", "Unknown"),
                "description": obj.get("description", "").replace("\n", " ")
            }

# 2) Load all malware names (for example contexts)
malware_map = {}
for fn in os.listdir(MALWARE_DIR):
    with open(os.path.join(MALWARE_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        if obj.get("type") == "malware":
            malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# 3) Gather "uses" examples per technique
examples = defaultdict(list)
for fn in os.listdir(RELATIONSHIP_DIR):
    with open(os.path.join(RELATIONSHIP_DIR, fn), encoding="utf-8") as f:
        bundle = json.load(f)
    for rel in bundle.get("objects", []):
        if (rel.get("type") == "relationship" and
            rel.get("relationship_type") == "uses"):
            mid = rel.get("source_ref")
            tid = rel.get("target_ref")
            desc = rel.get("description", "").strip()
            if mid in malware_map and tid in tech_map and desc:
                examples[tid].append(f"{malware_map[mid]}: {desc}")

# 4) Build one chunk per technique, including description + examples
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
        "name": info["name"]
    })

# 4.5) **Preview**: ghi tất cả chunks ra file để bạn kiểm tra
with open(PREVIEW_FILE, "w", encoding="utf-8") as out:
    for chunk in chunks:
        out.write(chunk)
        out.write("\n\n---\n\n")
print(f"📄 Đã ghi preview chunks vào: {PREVIEW_FILE}")

