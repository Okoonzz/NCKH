import json

def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_text_event_sequence(report):
    """
    Trả về danh sách chuỗi mô tả sự kiện, đã sắp theo timestamp tăng dần.
    Mỗi chuỗi: "<timestamp>:<EVENT_TYPE>:<detail>"
    """
    evts = []
    # thời gian mặc định khi event không có 'time'
    start_time = report.get('info', {}).get('started', 0.0)
    end_time   = report.get('info', {}).get('ended', start_time)

    # 1. API calls
    for proc in report.get('behavior', {}).get('processes', []):
        for call in proc.get('calls', []):
            t   = call.get('time', start_time)
            api = call.get('api', 'UNKNOWN')
            evts.append((t, f"{t:.3f}:API:{api}"))

    # 2. File / Registry / DLL từ behavior.generic
    for item in report.get('behavior', {}).get('generic', []):
        t       = item.get('first_seen', start_time)
        summary = item.get('summary', {})

        # File opened
        for path in summary.get('file_opened', []):
            evts.append((t, f"{t:.3f}:FILE_OPEN:{path}"))
        # Registry keys opened
        for key in summary.get('regkey_opened', []):
            evts.append((t, f"{t:.3f}:REGKEY_OPEN:{key}"))
        # Registry keys read
        for key in summary.get('regkey_read', []):
            evts.append((t, f"{t:.3f}:REGKEY_READ:{key}"))
        # DLLs loaded
        for dll in summary.get('dll_loaded', []):
            evts.append((t, f"{t:.3f}:DLL_LOAD:{dll}"))

    # 3. UDP/TCP
    for pkt in report.get('network', {}).get('udp', []):
        t    = pkt.get('time', start_time)
        evts.append((t, f"{t:.3f}:NET_UDP:{pkt.get('src')}->{pkt.get('dst')}:{pkt.get('dport')}"))
    for pkt in report.get('network', {}).get('tcp', []):
        t    = pkt.get('time', start_time)
        evts.append((t, f"{t:.3f}:NET_TCP:{pkt.get('src')}->{pkt.get('dst')}:{pkt.get('dport')}"))

    # 4. HTTP / HTTP_EX / HTTPS_EX
    for http in report.get('network', {}).get('http', []):
        t      = http.get('time', start_time)
        method = http.get('method', '')
        uri    = http.get('uri') or http.get('path', '')
        evts.append((t, f"{t:.3f}:NET_HTTP:{method} {uri}"))
    for ex in report.get('network', {}).get('http_ex', []):
        t      = ex.get('time', start_time)
        evts.append((t, f"{t:.3f}:NET_HTTP_EX:{ex.get('method','')} {ex.get('host','')}"))
    for ex in report.get('network', {}).get('https_ex', []):
        t      = ex.get('time', start_time)
        evts.append((t, f"{t:.3f}:NET_HTTPS_EX:{ex.get('host','')}"))

    # 5. DNS
    for dns in report.get('network', {}).get('dns', []):
        t    = dns.get('time', start_time)
        evts.append((t, f"{t:.3f}:NET_DNS:{dns.get('type','')} {dns.get('request','')}"))

    # 6. Dropped files (nếu có)
    for drop in report.get('dropped', []):
        t    = drop.get('time', end_time)
        name = drop.get('name', drop.get('dst', ''))
        evts.append((t, f"{t:.3f}:FILE_DROPPED:{name}"))

    # 7. Sắp xếp và trả về danh sách mô tả
    evts.sort(key=lambda x: x[0])
    return [desc for _, desc in evts]


if __name__ == '__main__':
    report = load_report('/home/xlstm/NCKH/features2.json')
    seq    = build_text_event_sequence(report)
    print(f"Total events: {len(seq)}\n")
    # Kiểm tra xem có các event REGKEY_OPEN không
    for e in seq:
        if 'REGKEY_OPEN' in e:
            print(e)
