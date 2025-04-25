import json

def extract_list(d, key):
    """
    Nếu d[key] tồn tại và là list thì trả về list đó,
    ngược lại trả về một list rỗng.
    """
    return d.get(key, []) if isinstance(d.get(key), list) else []

def main():
    # 1) Đọc report từ file report.json
    with open("report.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    features = {}

    # 2) Trích behavior summary
    summary = report.get("behavior", {}).get("summary", {})
    behavior_keys = [
        "file_created", "file_deleted", "file_read", "file_recreated", "file_exists",
        "file_opened",
        "regkey_opened", "regkey_read", "regkey_written",
        "dll_loaded", "executed_commands", "directory_created",
        "mutex", "processes_created", "resolves_host"
    ]
    features["behavior_summary"] = {
        k: extract_list(summary, k) for k in behavior_keys
    }

    # 3) Trích dropped files
    features["dropped_files"] = report.get("dropped", [])

    # 4) Trích thống kê API calls
    features["api_statistics"] = report.get("behavior", {}).get("apistats", {})

    # 5) Trích network activities
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

    # 6) Trích static imports
    static = report.get("static", {})
    imports = []
    for dll in static.get("pe_imports", []):
        for imp in dll.get("imports", []):
            name = imp.get("name")
            if name:
                imports.append(name)
    features["static_imports"] = imports

    # 7) Trích signatures → TTP mapping
    ttps = {}
    for sig in report.get("signatures", []):
        for ttp_id, ttp in sig.get("ttp", {}).items():
            ttps[ttp_id] = ttp.get("short", "")
    features["signatures_ttp"] = ttps

    # 8) Trích processes detail
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

    # 9) Ghi kết quả ra features.json
    with open("features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    print("Đã trích xuất đầy đủ features vào file features.json")

if __name__ == "__main__":
    main()
