# data/sequence_extractor.py

import json

def load_report(path):
    """
    Load JSON features file (features2.json) from the given path.
    Returns a dict containing all extracted features.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_text_event_sequence(report):
    """
    Từ dict `report` như cấu trúc của features2.json, tạo
    một danh sách các chuỗi event mô tả mọi feature.
    Mỗi event có định dạng "<timestamp>:<TYPE>:<detail>".
    Nếu feature không có timestamp riêng, gán timestamp = 0.0.
    
    Trả về list các chuỗi event, đã sắp xếp theo timestamp tăng dần.
    """
    evts = []
    t0 = 0.0

    # 1) behavior_summary
    for key, arr in report.get("behavior_summary", {}).items():
        typ = f"BEHAVIOR_{key.upper()}"
        for item in arr:
            evts.append((t0, f"{t0:.3f}:{typ}:{item}"))

    # 2) dropped_files
    for d in report.get("dropped_files", []):
        t    = d.get("time", t0)
        name = d.get("name", d.get("dst", ""))
        evts.append((t, f"{t:.3f}:FILE_DROPPED:{name}"))

    # 3) api_call_sequence
    for call in report.get("api_call_sequence", []):
        t       = call.get("time", t0)
        api     = call.get("api", "")
        pid     = call.get("pid", "")
        cat     = call.get("category", "")
        args    = call.get("arguments", {})
        detail  = f"{api}|pid={pid}|cat={cat}"
        # Thêm các arg quan trọng nếu có
        for k in ("filename","filepath","regkey","command","host","ip_address"):
            if k in args:
                detail += f"|{k}={args[k]}"
        evts.append((t, f"{t:.3f}:API:{detail}"))

    # 4) network activity
    net = report.get("network", {})
    # HTTP
    for h in net.get("http", []):
        t      = h.get("time", t0)
        method = h.get("method", "")
        uri    = h.get("uri", h.get("path", ""))
        evts.append((t, f"{t:.3f}:NET_HTTP:{method} {uri}"))
    # TCP/UDP/ICMP/SMTP
    for proto in ("tcp","udp","icmp","smtp"):
        for pkt in net.get(proto, []):
            t     = pkt.get("time", t0)
            s     = pkt.get("sport", "")
            d     = pkt.get("dport", "")
            evts.append((t, f"{t:.3f}:NET_{proto.upper()}:{s}->{d}"))
    # DNS
    for pkt in net.get("dns", []):
        t   = pkt.get("time", t0)
        req = pkt.get("request", "")
        evts.append((t, f"{t:.3f}:NET_DNS:{req}"))
    # Hosts
    for host in net.get("hosts", []):
        evts.append((t0, f"{t0:.3f}:NET_HOST:{host}"))

    # 5) static_imports
    for imp in report.get("static_imports", []):
        evts.append((t0, f"{t0:.3f}:STATIC_IMPORT:{imp}"))

    # 6) signatures_ttp
    for ttp_id in report.get("signatures_ttp", {}).keys():
        evts.append((t0, f"{t0:.3f}:SIG_TTP:{ttp_id}"))

    # 7) processes (name + command line)
    for p in report.get("processes", []):
        t    = p.get("first_seen", t0)
        name = p.get("name", "")
        cmd  = p.get("cmdline", "")
        evts.append((t, f"{t:.3f}:PROC_NAME:{name}"))
        if cmd:
            evts.append((t, f"{t:.3f}:PROC_CMD:{cmd}"))

    # 8) Sort theo timestamp và trả danh sách text
    evts.sort(key=lambda x: x[0])
    return [desc for _, desc in evts]


if __name__ == "__main__":
    rpt = load_report("/home/xlstm/NCKH/report.json")
    seq = build_text_event_sequence(rpt)
    for line in seq:
        print(line)
