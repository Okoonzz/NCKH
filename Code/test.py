import os
import json

BUNDLE_DIR = "cti/enterprise-attack/malware"
output_path = "data/mitre_malware.txt"

if not os.path.exists("data"):
    os.mkdir("data")

with open(output_path, "w", encoding="utf-8") as out:
    for filename in os.listdir(BUNDLE_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(BUNDLE_DIR, filename), "r", encoding="utf-8") as f:
                bundle = json.load(f)
                if "objects" not in bundle:
                    continue  # skip nếu không đúng định dạng STIX Bundle

                for obj in bundle["objects"]:
                    if obj.get("type") == "malware":
                        name = obj.get("name", "Unknown Malware")
                        description = obj.get("description", "Không có mô tả.")
                        aliases = ", ".join(obj.get("x_mitre_aliases", []))
                        platforms = ", ".join(obj.get("x_mitre_platforms", []))
                        external_id = next((
                            ref.get("external_id") for ref in obj.get("external_references", [])
                            if ref.get("source_name") == "mitre-attack"
                        ), "N/A")
                        references = [
                            f"- {ref.get('source_name', '')}: {ref.get('url', '')}"
                            for ref in obj.get("external_references", [])
                            if "url" in ref
                        ]

                        out.write(f"Tên Malware: {name}\n")
                        out.write(f"Mã định danh: {external_id}\n")
                        out.write(f"Alias: {aliases}\n")
                        out.write(f"Nền tảng: {platforms}\n")
                        out.write(f"Mô tả: {description}\n")
                        if references:
                            out.write("Nguồn tham khảo:\n")
                            out.write("\n".join(references) + "\n")
                        out.write("---\n\n")
