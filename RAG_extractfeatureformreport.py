import json

def extract_list(d, key):
    return d.get(key, []) if isinstance(d.get(key), list) else []

def main():
    with open("report.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    features = {}

    # 1) Behavior summary
    summary = report.get("behavior", {}).get("summary", {})
    behavior_keys = [
        "file_created", "file_deleted", "file_read", "file_recreated", "file_exists",
        "file_opened", "regkey_opened", "regkey_read", "regkey_written",
        "dll_loaded", "executed_commands", "directory_created",
        "mutex", "processes_created", "resolves_host"
    ]
    features["behavior_summary"] = {k: extract_list(summary, k) for k in behavior_keys}

    # 2) Dropped files
    features["dropped_files"] = report.get("dropped", [])

    # 3) API statistics
    features["api_statistics"] = report.get("behavior", {}).get("apistats", {})

    # 4) API call sequence (ordered)
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
    # Sort theo thời gian
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

    # 7) Signatures - TTP only
    ttps = {}
    for sig in report.get("signatures", []):
        for ttp_id, ttp in sig.get("ttp", {}).items():
            ttps[ttp_id] = ttp.get("short", "")
    features["signatures_ttp"] = ttps

    # 8) Signatures - Full
    sig_full = []
    for sig in report.get("signatures", []):
        sig_full.append({
            "name": sig.get("name"),
            "description": sig.get("description"),
            "severity": sig.get("severity"),
            "ttp": sig.get("ttp", {}),
            "markcount": sig.get("markcount", 0),
            "marks": sig.get("marks", [])
        })
    features["signatures_full"] = sig_full

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

    # 10) Save output
    with open("features2.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    print("✅ Đã trích xuất đầy đủ features vào file features2.json")

if __name__ == "__main__":
    main()
