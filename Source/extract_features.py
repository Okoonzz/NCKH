import json
import os

def extract_list(d, key):
    return d.get(key, []) if isinstance(d.get(key), list) else []

def extract_features(report):
    features = {}

    # 0) Static hashes
    # target_file = report.get("target", {}).get("file", {})
    # features["hashes_sample"] = {
    #     "sha1": target_file.get("sha1"),
    #     "sha256": target_file.get("sha256"),
    #     "sha512": target_file.get("sha512"),
    #     "md5": target_file.get("md5"),
    #     "crc32": target_file.get("crc32"),
    # }

    # 1) Behavior summary
    summary = report.get("behavior", {}).get("summary", {})
    # behavior_keys = [
    #     "file_created", "file_deleted", "file_read", "file_recreated", "file_exists",
    #     "file_opened", "regkey_opened", "regkey_read", "regkey_written",
    #     "dll_loaded", "executed_commands", "directory_created",
    #     "mutex", "processes_created", "resolves_host"
    # ]
    features["behavior_summary"] = summary

    # 2) Dropped files
    features["dropped_files"] = report.get("dropped", [])

    # 3) API statistics
    # features["api_statistics"] = report.get("behavior", {}).get("apistats", {})

    # 4) API call sequence
    api_calls = []
    for proc in report.get("behavior", {}).get("processes", []):
        for call in proc.get("calls", []):
            api_calls.append({
                "time": call.get("time"),
                "api": call.get("api"),
                "pid": proc.get("pid"),
                "arguments": call.get("arguments", {}),
                "category": call.get("category", "")
            })
    api_calls.sort(key=lambda x: x["time"])
    features["api_call_sequence"] = api_calls

    # 5) Network activity
    net = report.get("network", {})
    features["network"] = {
        "http": net.get("http", []),
        "tcp": net.get("tcp", []),
        "udp": net.get("udp", []),
        "dns": net.get("dns", []),
        "hosts": net.get("hosts", []),
        "icmp": net.get("icmp", []),
        "smtp": net.get("smtp", [])
    }

    # 6) Static imports
    static = report.get("static", {})
    imports = []
    for dll in static.get("pe_imports", []):
        for imp in dll.get("imports", []):
            name = imp.get("name")
            if name:
                imports.append(name)
    features["static_imports"] = imports

    # 7) Unified Signatures (gộp cả TTP & full)
    features["signatures"] = []
    for sig in report.get("signatures", []):
        features["signatures"].append({
            "node_type":   "Signature",
            "name":        sig.get("name", ""),
            "description": sig.get("description", ""),
            "severity":    sig.get("severity", 0),
            "ttp_ids":     list(sig.get("ttp", {}).keys()),
            "markcount":   sig.get("markcount", 0),
            "marks":       sig.get("marks", [])
        })


    # 9) Process details
    processes = []
    for p in report.get("behavior", {}).get("processes", []):
        processes.append({
            "name": p.get("process_name"),
            "path": p.get("process_path"),
            "pid": p.get("pid"),
            "cmdline": p.get("command_line"),
            "modules": [
                m.get("basename")
                for m in p.get("modules", [])
                if m.get("basename")
            ]
        })
    features["processes"] = processes

    return features

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                    report = json.load(f)

                # Kiểm tra nếu KHÔNG có bất kỳ calls nào trong processes
                has_calls = False
                for proc in report.get("behavior", {}).get("processes", []):
                    if proc.get("calls"):
                        has_calls = True
                        break

                # if not has_calls:
                #    print(f"⏭️  Bỏ qua {filename} (không có API calls)")
                #    continue

                features = extract_features(report)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(features, f, indent=2, ensure_ascii=True)

                print(f"✅ Đã xử lý: {filename}")
            except Exception as e:
                print(f"❌ Lỗi với {filename}: {e}")


def main():
    process_folder("benign-test", "reports_test/benign")
    #process_folder("/home/xlstm/work/cuckoo_benign_reports", "reports/benign")

if __name__ == "__main__":
    main()
# def process_from_contexts(a_dir, b_dir, c_dir):
#     os.makedirs(c_dir, exist_ok=True)
#     for filename in os.listdir(a_dir):
#         if filename.endswith("_contexts.json"):
#             base_name = filename.replace("_contexts.json", ".json")
#             b_file_path = os.path.join(b_dir, base_name)
#             c_file_path = os.path.join(c_dir, base_name)

#             if not os.path.isfile(b_file_path):
#                 print(f"⚠️  Không tìm thấy file {base_name} trong b/")
#                 continue

#             try:
#                 with open(b_file_path, "r", encoding="utf-8", errors="replace") as f:
#                     report = json.load(f)

#                 features = extract_features(report)

#                # with open(c_file_path, "w", encoding="utf-8") as f:
#                 #    json.dump(features, f, indent=2, ensure_ascii=True)

#                 #print(f"✅ Đã xử lý: {base_name}")
#                 # Chỉ lưu nếu có api_call_sequence không rỗng
#                 if not features.get("api_call_sequence"):
#                     print(f"⏭️  Bỏ qua {base_name} (api_call_sequence rỗng)")
#                     continue
#             except Exception as e:
#                 print(f"❌ Lỗi với {base_name}: {e}")

#def main():
#    process_from_contexts("/home/xlstm/Downloads/little/context_MS", "cuckoo_reports/", "a/")

#if __name__ == "__main__":
#    main()