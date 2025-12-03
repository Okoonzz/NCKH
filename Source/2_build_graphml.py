import os
import glob
import json
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_ransomware_graph(report_path: str) -> nx.DiGraph:
    # 1) Load report only
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    G = nx.DiGraph()

    # 2) Feature nodes
    for ft, vals in report.get('behavior_summary', {}).items():
        for v in vals:
            nid = f"feature:{ft}:{v}"
            G.add_node(nid,
                       node_type='feature',
                       feature_type=ft,
                       feature_value=v)

    # 3) API call nodes
    api_nodes = []
    api_sequence = report.get('api_call_sequence', [])[:1500]  # Limit to first 1500 calls for performance
    for idx, call in enumerate(api_sequence):
        nid = f"api:{idx}:{call['api']}"
        G.add_node(nid,
                   node_type='api',
                   api=call['api'],
                   pid=call['pid'],
                   time=call['time'],
                   arguments=json.dumps(call.get('arguments', {})))
        api_nodes.append((nid, call))

    # 4) Sequence edges
    for (n1, _), (n2, _) in zip(api_nodes, api_nodes[1:]):
        G.add_edge(n1, n2, relation='sequence')

    # 5) Feature → API edges
    for nid, call in api_nodes:
        args = call.get('arguments', {})
        for ft, vals in report.get('behavior_summary', {}).items():
            # print(f"Processing feature {ft}")
            for v in vals:
                if any(v == str(x) for x in args.values()):
                    G.add_edge(f"feature:{ft}:{v}", nid, relation='uses')

    # 6) Process nodes & calls edges
    for p in report.get('processes', []):
        pid = p['pid']
        pn = f"process:{pid}"
        G.add_node(pn,
                   node_type='process',
                   pid=pid,
                   #name=p.get('name',''),
                   process_name = p.get('name',''),
                   path=p.get('path',''),
                   cmdline=p.get('cmdline',''))
        for nid, call in api_nodes:
            if call['pid'] == pid:
                G.add_edge(pn, nid, relation='calls')

    # 7) Dropped-file nodes & edges
    for d in report.get('dropped_files', []):
        path = d.get('filepath', d.get('path', ''))
        dn = f"dropped:{path}"
        G.add_node(dn,
                   node_type='dropped_file',
                   filepath=path,
                   md5=d.get('md5',''),
                   size=d.get('size',''))
        for pid in d.get('pids', []):
            pn = f"process:{pid}"
            if pn in G:
                G.add_edge(pn, dn, relation='dropped')

    # 8) Network nodes & time-based correlation
    # Trước đó ở đầu file, định nghĩa whitelist các network‐API phổ biến:
    NETWORK_APIS = {
        "socket", "connect", "send", "sendto", "recv", "recvfrom",
        "WSASocketA", "WSAConnect", "HttpOpenRequestA", "InternetConnectA",
        "SSL_connect", "BIO_read", "BIO_write"
    }

    # --- 8) Network nodes & smart correlation ---
    for proto, entries in report.get('network', {}).items():
        for idx, entry in enumerate(entries):
            nn = f"network:{proto}:{idx}"
            # 1) Tạo node network
            G.add_node(nn,
                       node_type='network',
                       category=proto,
                       details=json.dumps(entry))

            # 2) Bỏ qua nếu không có timestamp
            if not (isinstance(entry, dict) and 'time' in entry):
                continue
            evt_time = entry['time']

            # 3) Lấy thông tin target IP/port (nếu có)
            target_ip   = entry.get('dst') or entry.get('dst_ip')
            target_port = entry.get('dst_port') or entry.get('port')
            #print(f"Processing network event {nn} with target_ip={target_ip}, target_port={target_port}, time={evt_time}")

            # 4) Step 1: lọc theo IP/port trong arguments
            matched = []
            for nid, call in api_nodes:
                # if call['time'] > evt_time:
                #     continue
                args = call.get('arguments', {})
                if target_ip and any(str(val) == str(target_ip) for val in args.values()):
                    matched.append((nid, call['time'], call['pid']))
                    print(f"Matched by IP: network event {nn} matched API call {nid} with args {args}")
                elif target_port and any(str(val) == str(target_port) for val in args.values()):
                    matched.append((nid, call['time'], call['pid']))

            # 5) Step 2: nếu không có match IP/port, lọc theo whitelist network‐API
            if not matched:
                for nid, call in api_nodes:
                    if call['time'] <= evt_time and call['api'] in NETWORK_APIS:
                        matched.append((nid, call['time'], call['pid']))

            if not matched:
                continue

            sel_api, _, sel_pid = max(matched, key=lambda x: x[1])
            print(matched, sel_api, sel_pid)

            G.add_edge(sel_api, nn, relation='connects')
            print(f"Linked network event {nn} to API call {sel_api}")
            pn = f"process:{sel_pid}"
            if pn in G:
                G.add_edge(pn, nn, relation='networked')


    # 9) Signature nodes & triggers
    for sig in report.get('signatures', []):
        name = sig['name']
        sn = f"signature:{name}"
        G.add_node(sn,
                   node_type='signature',
                   signature_name=name,
                   description=sig.get('description',''),
                   severity=sig.get('severity',''))
        for mark in sig.get('marks', []):
            pid = mark.get('pid')
            pn = f"process:{pid}"
            if pn in G:
                G.add_edge(sn, pn, relation='sig_on_proc')
            call = mark.get('call', {})
            for nid, c in api_nodes:
                if (c['pid'], c['time'], c['api']) == (
                        pid, call.get('time'), call.get('api')):
                    G.add_edge(sn, nid, relation='triggers')


    return G


def sanitize_string(s: str) -> str:
    """
    Loại bỏ mọi ký tự không nằm trong tập hợp hợp lệ của XML:
      - Giữ lại: U+0009 (TAB), U+000A (LF), U+000D (CR)
      - Hoặc mã từ U+0020 đến U+D7FF
      - Hoặc mã từ U+E000 đến U+FFFD
      - Hoặc mã từ U+10000 đến U+10FFFF

    Tất cả ký tự khác (như 0x00–0x08, 0x0B–0x0C, 0x0E–0x1F, 0x7F, hay các surrogate không khớp) đều bị loại bỏ.
    """
    out_chars = []
    for ch in s:
        code = ord(ch)
        # Nếu là TAB, LF, CR
        if code == 0x9 or code == 0xA or code == 0xD:
            out_chars.append(ch)
        # Nếu trong khoảng U+0020 … U+D7FF
        elif 0x20 <= code <= 0xD7FF:
            out_chars.append(ch)
        # Nếu trong khoảng U+E000 … U+FFFD
        elif 0xE000 <= code <= 0xFFFD:
            out_chars.append(ch)
        # Nếu trong khoảng U+10000 … U+10FFFF
        elif 0x10000 <= code <= 0x10FFFF:
            out_chars.append(ch)
        # Ngược lại: bỏ qua (control char, NULL byte, surrogate sai, v.v.)
        else:
            continue
    return ''.join(out_chars)

def sanitize_and_save_graph(G: nx.DiGraph, out_path: str):
    """
    1) Với mỗi node gốc trong G, tạo một ID mới (chuỗi) đã được sanitize (sanitize_string).
       Nếu sanitize_string(node_id) trả về chuỗi trống hoặc trùng lặp, sẽ tự động thêm hậu tố _1, _2,…
    2) Sao chép từng thuộc tính (attribute) của node, ép thành str/json rồi sanitize_string chúng.
    3) Tương tự với từng edge: sanitize cả edge key và edge value.
    4) Cuối cùng, ghi ra file .graphml bằng nx.write_graphml với đồ thị mới (đã sạch).
    """
    # --- 2.1) Xây mapping từ old_node_id -> new_node_id (đã sanitize)
    mapping = {}
    used_ids = set()

    for orig in G.nodes():
        # 2.1.1) sanitize chuỗi orig (orig bắt buộc str)
        orig_str = str(orig)
        sanitized = sanitize_string(orig_str)

        # Nếu sanitize xong ra chuỗi rỗng, đặt tên mặc định "node"
        if sanitized == "":
            sanitized = "node"

        # 2.1.2) Tránh trùng lặp: nếu đã có trong used_ids thì thêm hậu tố "_1", "_2", ...
        candidate = sanitized
        suffix = 1
        while candidate in used_ids:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1

        used_ids.add(candidate)
        mapping[orig] = candidate

    # --- 2.2) Tạo đồ thị mới H với node ID đã sanitize và attribute cũng sanitize
    H = nx.DiGraph()

    # 2.2.1) Copy nodes
    for orig, new_id in mapping.items():
        attrs = G.nodes[orig]
        clean_attrs = {}

        for k, v in attrs.items():
            # Bước a: ép v thành chuỗi (JSON hoặc str) trước
            if not isinstance(v, (str, int, float, bool)):
                try:
                    v_str = json.dumps(v, ensure_ascii=False)
                except Exception:
                    v_str = str(v)
            else:
                v_str = str(v)

            # Bước b: sanitize từng ký tự trong chuỗi đó
            clean_attrs[k] = sanitize_string(v_str)

        H.add_node(new_id, **clean_attrs)

    # 2.2.2) Copy edges
    for u, v, edge_data in G.edges(data=True):
        new_u = mapping[u]
        new_v = mapping[v]

        clean_edge_data = {}
        for k, w in edge_data.items():
            # Ep w về chuỗi
            if not isinstance(w, (str, int, float, bool)):
                try:
                    w_str = json.dumps(w, ensure_ascii=False)
                except Exception:
                    w_str = str(w)
            else:
                w_str = str(w)

            # Sanitize
            clean_edge_data[k] = sanitize_string(w_str)

        H.add_edge(new_u, new_v, **clean_edge_data)

    # --- 2.3) Cuối cùng, ghi H ra file .graphml
    nx.write_graphml(H, out_path)


def process_single_report(args):
    """
    Xử lý 1 file report JSON:
      1) build graph
      2) sanitize attributes
      3) lưu .graphml
    Nhận vào args = (report_path, out_dir).
    """
    report_path, out_dir = args
    basename = os.path.splitext(os.path.basename(report_path))[0]
    try:
        G = build_ransomware_graph(report_path)
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            G.remove_nodes_from(isolated_nodes)
        out_path = os.path.join(out_dir, f"{basename}.graphml")
        sanitize_and_save_graph(G, out_path)
        return f"OK: {basename}"
    except Exception as e:
        return f"ERROR: {basename} → {e}"

def batch_build_graphs_multithread(
    reports_dir='data_ransomware_extracted2025/ransomware',
    #reports_dir='2025_103_extracted/ransomware',
    out_dir='graph_ransomware2025_reports/ransomware',
    max_workers=None
):
    """
    Hàm chính để chạy song song đa luồng:
    - reports_dir: thư mục chứa các file .json
    - out_dir: thư mục sẽ lưu kết quả .graphml
    - max_workers: số luồng tối đa (mặc định 8)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Tìm tất cả file JSON
    report_paths = glob.glob(os.path.join(reports_dir, '*.json'))
    if not report_paths:
        print(f"[!] Không tìm thấy file JSON nào trong {reports_dir}")
        return

    # 2) Chuẩn bị args cho từng luồng
    tasks = [(rp, out_dir) for rp in report_paths]

    # 3) Dùng ThreadPoolExecutor để submit song song
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_report = {
            executor.submit(process_single_report, args): args[0]
            for args in tasks
        }

        # 4) Khi mỗi luồng hoàn tất, in kết quả
        for future in as_completed(future_to_report):
            report_file = os.path.basename(future_to_report[future])
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"ERROR ON {report_file}: {exc}")

if __name__ == '__main__':
    # Muốn tăng/giảm số luồng, chỉ cần sửa max_workers.
    # Ví dụ: max_workers=4, max_workers=None (Python sẽ tự chọn số luồng = CPU count).
    batch_build_graphs_multithread(max_workers=None)
