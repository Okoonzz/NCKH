# import os
# import json

# MALWARE_DIR = "cti/enterprise-attack/malware"
# RELATIONSHIP_DIR = "cti/enterprise-attack/relationship"
# ATTACK_PATTERN_DIR = "cti/enterprise-attack/attack-pattern"
# OUTPUT_PATH = "data/malware_with_techniques.txt"

# # === Bước 1: Load attack-patterns ===
# attack_pattern_map = {}
# for fname in os.listdir(ATTACK_PATTERN_DIR):
#     with open(os.path.join(ATTACK_PATTERN_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "attack-pattern":
#                 attack_pattern_map[obj["id"]] = {
#                     "name": obj.get("name"),
#                     "external_id": next((
#                         ref.get("external_id") for ref in obj.get("external_references", [])
#                         if ref.get("source_name") == "mitre-attack"
#                     ), None)
#                 }

# # === Bước 2: Load relationships (malware → technique + use desc) ===
# malware_to_techniques = {}
# for fname in os.listdir(RELATIONSHIP_DIR):
#     with open(os.path.join(RELATIONSHIP_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "relationship" and obj.get("relationship_type") == "uses":
#                 src = obj.get("source_ref")
#                 tgt = obj.get("target_ref")
#                 desc = obj.get("description", "")
#                 if src.startswith("malware--") and tgt.startswith("attack-pattern--"):
#                     malware_to_techniques.setdefault(src, []).append({
#                         "tech_id": tgt,
#                         "description": desc
#                     })

# # === Bước 3: Ghi ra file ===
# if not os.path.exists("data"):
#     os.makedirs("data")

# with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
#     for fname in os.listdir(MALWARE_DIR):
#         with open(os.path.join(MALWARE_DIR, fname), "r", encoding="utf-8") as f:
#             bundle = json.load(f)
#             for obj in bundle.get("objects", []):
#                 if obj.get("type") != "malware":
#                     continue

#                 m_id = obj["id"]
#                 m_name = obj.get("name", "Unknown Malware")
#                 m_desc = obj.get("description", "Không có mô tả.")
#                 m_external_id = next((
#                     ref.get("external_id") for ref in obj.get("external_references", [])
#                     if ref.get("source_name") == "mitre-attack"
#                 ), "N/A")

#                 out.write(f"Tên malware: {m_name}\n")
#                 out.write(f"Mã định danh: {m_external_id}\n")
#                 out.write(f"Mô tả: {m_desc}\n\n")

#                 uses = malware_to_techniques.get(m_id, [])
#                 if uses:
#                     out.write("Kỹ thuật sử dụng:\n")
#                     for item in uses:
#                         tech_id = item["tech_id"]
#                         desc = item["description"]
#                         tech_info = attack_pattern_map.get(tech_id)
#                         if tech_info:
#                             out.write(f"- {tech_info['external_id']}: {tech_info['name']}\n")
#                             if desc:
#                                 out.write(f"  Use: {desc}\n")
#                 else:
#                     out.write("Không có kỹ thuật nào được ghi nhận.\n")
#                 out.write("\n---\n\n")

# print(f"✅ Đã ghi dữ liệu vào: {OUTPUT_PATH}")




######################### REV



# # === Đường dẫn đến CTI data ===
# OUTPUT_PATH = "data/technique_to_malware.txt"

# # === Load toàn bộ malware ===
# malware_map = {}  # id -> name
# for fname in os.listdir(MALWARE_DIR):
#     with open(os.path.join(MALWARE_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "malware":
#                 malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# # === Load attack-patterns ===
# attack_pattern_info = {}  # id -> (id, name)
# for fname in os.listdir(ATTACK_PATTERN_DIR):
#     with open(os.path.join(ATTACK_PATTERN_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "attack-pattern":
#                 attack_pattern_info[obj["id"]] = {
#                     "name": obj.get("name", ""),
#                     "external_id": next((
#                         ref.get("external_id") for ref in obj.get("external_references", [])
#                         if ref.get("source_name") == "mitre-attack"
#                     ), None)
#                 }

# # === Gom theo kỹ thuật → malware ===
# technique_to_malware = {}

# for fname in os.listdir(RELATIONSHIP_DIR):
#     with open(os.path.join(RELATIONSHIP_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "relationship" and obj.get("relationship_type") == "uses":
#                 src = obj.get("source_ref")
#                 tgt = obj.get("target_ref")
#                 desc = obj.get("description", "")
#                 if src.startswith("malware--") and tgt.startswith("attack-pattern--"):
#                     malware_name = malware_map.get(src)
#                     technique_to_malware.setdefault(tgt, []).append({
#                         "malware": malware_name,
#                         "use": desc
#                     })

# # === Ghi ra file technique_to_malware.txt ===
# if not os.path.exists("data"):
#     os.makedirs("data")

# with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
#     for tid, malware_list in technique_to_malware.items():
#         t_info = attack_pattern_info.get(tid, {})
#         out.write(f"Kỹ thuật: {t_info.get('name', 'Unknown')} ({t_info.get('external_id', tid)})\n")
#         out.write("Được sử dụng bởi:\n")
#         for m in malware_list:
#             out.write(f"- {m['malware']}\n")
#             if m['use']:
#                 out.write(f"  Use: {m['use']}\n")
#         out.write("\n---\n\n")

# print("✅ Đã tạo xong file data/technique_to_malware.txt")






# MALWARE_DIR = "cti/enterprise-attack/malware"
# RELATIONSHIP_DIR = "cti/enterprise-attack/relationship"
# ATTACK_PATTERN_DIR = "cti/enterprise-attack/attack-pattern"
# OUTPUT_FILE = "data/malware_use_chunks.txt"

# # Load malware id → name
# malware_map = {}
# for fname in os.listdir(MALWARE_DIR):
#     with open(os.path.join(MALWARE_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "malware":
#                 malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# # Load technique id → name + ID
# tech_map = {}
# for fname in os.listdir(ATTACK_PATTERN_DIR):
#     with open(os.path.join(ATTACK_PATTERN_DIR, fname), "r", encoding="utf-8") as f:
#         bundle = json.load(f)
#         for obj in bundle.get("objects", []):
#             if obj.get("type") == "attack-pattern":
#                 tech_map[obj["id"]] = {
#                     "name": obj.get("name", "Unknown"),
#                     "external_id": next((ref.get("external_id") for ref in obj.get("external_references", [])
#                                         if ref.get("source_name") == "mitre-attack"), "N/A")
#                 }

# # Duyệt relationship để tạo đoạn use
# with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
#     for fname in os.listdir(RELATIONSHIP_DIR):
#         with open(os.path.join(RELATIONSHIP_DIR, fname), "r", encoding="utf-8") as f:
#             bundle = json.load(f)
#             for rel in bundle.get("objects", []):
#                 if rel.get("type") == "relationship" and rel.get("relationship_type") == "uses":
#                     malware_id = rel.get("source_ref")
#                     tech_id = rel.get("target_ref")
#                     use_desc = rel.get("description", "").strip()
                    
#                     if not use_desc:
#                         continue

#                     malware_name = malware_map.get(malware_id)
#                     tech_info = tech_map.get(tech_id)
#                     if malware_name and tech_info:
#                         out.write(f"Malware: {malware_name}\n")
#                         out.write(f"Technique: {tech_info['name']} ({tech_info['external_id']})\n")
#                         out.write(f"Use: {use_desc}\n")
#                         out.write("\n---\n\n")

# print("✅ Đã tạo file 'data/malware_use_chunks.txt'")



import os
import json

# === Cấu hình thư mục ===
MALWARE_DIR = "cti/enterprise-attack/malware"
RELATIONSHIP_DIR = "cti/enterprise-attack/relationship"
ATTACK_PATTERN_DIR = "cti/enterprise-attack/attack-pattern"
OUTPUT_FILE = "data/technique_use_chunks.txt"

# === 1. Load malware ID → name ===
malware_map = {}
for fname in os.listdir(MALWARE_DIR):
    with open(os.path.join(MALWARE_DIR, fname), "r", encoding="utf-8") as f:
        bundle = json.load(f)
        for obj in bundle.get("objects", []):
            if obj.get("type") == "malware":
                malware_map[obj["id"]] = obj.get("name", "Unknown Malware")

# === 2. Load technique ID → {name, external_id} ===
tech_map = {}
for fname in os.listdir(ATTACK_PATTERN_DIR):
    with open(os.path.join(ATTACK_PATTERN_DIR, fname), "r", encoding="utf-8") as f:
        bundle = json.load(f)
        for obj in bundle.get("objects", []):
            if obj.get("type") == "attack-pattern":
                tech_map[obj["id"]] = {
                    "name": obj.get("name", "Unknown"),
                    "external_id": next((
                        ref.get("external_id")
                        for ref in obj.get("external_references", [])
                        if ref.get("source_name") == "mitre-attack"
                    ), "N/A")
                }

# === 3. Xây dựng technique → [malware + use] ===
tech_to_use = {}

for fname in os.listdir(RELATIONSHIP_DIR):
    with open(os.path.join(RELATIONSHIP_DIR, fname), "r", encoding="utf-8") as f:
        bundle = json.load(f)
        for rel in bundle.get("objects", []):
            if rel.get("type") == "relationship" and rel.get("relationship_type") == "uses":
                malware_id = rel.get("source_ref")
                tech_id = rel.get("target_ref")
                use_desc = rel.get("description", "").strip()

                if not use_desc:
                    continue

                malware_name = malware_map.get(malware_id)
                tech_info = tech_map.get(tech_id)
                if malware_name and tech_info:
                    key = tech_id
                    tech_to_use.setdefault(key, {
                        "name": tech_info["name"],
                        "external_id": tech_info["external_id"],
                        "uses": []
                    })["uses"].append((malware_name, use_desc))

# === 4. Ghi ra file `technique_use_chunks.txt` ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for tid, info in tech_to_use.items():
        out.write(f"Technique: {info['name']} ({info['external_id']})\n")
        for malware_name, use in info["uses"]:
            out.write(f"Used by: {malware_name}\n")
            out.write(f"Use: {use}\n\n")
        out.write("---\n\n")

print(f"✅ Đã tạo file '{OUTPUT_FILE}' thành công.")
