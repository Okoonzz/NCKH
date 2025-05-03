import json
import csv
import glob
import os

def extract_list(d, key):
    """
    Nếu d[key] tồn tại và là list thì trả về list đó,
    ngược lại trả về một list rỗng.
    """
    return d.get(key, []) if isinstance(d.get(key), list) else []


def stringify_list(lst):
    """
    Chuyển các phần tử trong list thành chuỗi;
    nếu phần tử là dict thì thử trích 'path' hoặc 'filepath',
    không thì dump JSON.
    """
    result = []
    for item in lst:
        if isinstance(item, dict):
            if 'path' in item:
                result.append(item['path'])
            elif 'filepath' in item:
                result.append(item['filepath'])
            else:
                result.append(json.dumps(item, ensure_ascii=False))
        else:
            result.append(str(item))
    return result


def extract_signature_features(report, features):
    """
    Trích xuất toàn bộ thông tin từ phần signatures vào features dict.
    """
    sigs = report.get("signatures", [])
    # Tên, mô tả, severity
    features['signature_names']      = ";".join([s.get('name','') for s in sigs])
    features['signature_descs']      = ";".join([s.get('description','') for s in sigs])
    features['signature_severities'] = ";".join([str(s.get('severity',0)) for s in sigs])
    # TTP IDs
    all_ttps = []
    for s in sigs:
        all_ttps += list(s.get('ttp', {}).keys())
    features['signature_ttps'] = ";".join(all_ttps)
    # Markcounts per signature and total count
    features['signature_markcounts'] = ";".join([str(s.get('markcount',0)) for s in sigs])
    features['count_signatures']     = len(sigs)
    # Phân tích marks: đếm theo category và gom API
    cat_counter = {}
    apis = []
    for s in sigs:
        for m in s.get('marks', []):
            cat = m.get('category') or m.get('call',{}).get('category') or 'unknown'
            cat_counter[cat] = cat_counter.get(cat,0) + 1
            if 'call' in m and isinstance(m['call'], dict):
                api = m['call'].get('api')
                if api:
                    apis.append(api)
    for cat, cnt in cat_counter.items():
        features[f'count_marks_{cat}'] = cnt
    features['signature_mark_apis'] = ";".join(apis)


def extract_features(report):
    features = {}

    # 1) Thông tin cơ bản của file
    target = report.get('target', {}).get('file', {})
    features['sha256']    = target.get('sha256','')
    features['md5']       = target.get('md5','')
    features['file_name'] = target.get('name','')

    # 2) Signature-level features
    extract_signature_features(report, features)

    # 3) Behavior summary
    summary = report.get('behavior', {}).get('summary', {})
    behavior_keys = [
        'file_created','file_deleted','file_read','file_recreated','file_exists',
        'file_opened','regkey_opened','regkey_read','regkey_written',
        'dll_loaded','executed_commands','directory_created',
        'mutex','processes_created','resolves_host'
    ]
    for k in behavior_keys:
        lst = extract_list(summary, k)
        features[f'count_{k}'] = len(lst)
        features[k] = ";".join(lst)

    # 4) Dropped files
    dropped = report.get('dropped', []) + report.get('dropped_files', [])
    features['count_dropped_files'] = len(dropped)
    features['dropped_files'] = ";".join(stringify_list(dropped))

    # 5) API statistics (15 APIs trọng yếu)
    apistats = report.get('behavior', {}).get('apistats', {})
    selected_apis = [
        'CreateFile','WriteFile','DeleteFile','CopyFile','MoveFile',
        'SetFilePointerEx','FlushFileBuffers','CryptEncrypt','CryptCreateHash','CryptHashData',
        'VirtualAlloc','VirtualProtect','NtAllocateVirtualMemory','NtProtectVirtualMemory','MapViewOfFile'
    ]
    for api in selected_apis:
        total = sum(pid_calls.get(api,0) for pid_calls in apistats.values())
        features[f'count_api_{api}'] = total

    # 6) Network activities
    net = report.get('network', {})
    features['count_domains']   = len(net.get('domains', []))
    features['domains']         = ";".join([d.get('domain','') for d in net.get('domains', [])])
    features['count_hosts']     = len(net.get('hosts', []))
    features['hosts']           = ";".join(net.get('hosts', []))
    features['count_http']      = len(net.get('http', []))
    features['net_http']        = ";".join([r.get('uri','') for r in net.get('http', [])])
    features['count_tcp']       = len(net.get('tcp', []))
    features['net_tcp']         = ";".join([f"{t.get('src')}->{t.get('dst')}:{t.get('dport')}" for t in net.get('tcp', [])])
    features['count_udp']       = len(net.get('udp', []))
    features['net_udp']         = ";".join([f"{u.get('src')}->{u.get('dst')}:{u.get('dport')}" for u in net.get('udp', [])])
    features['count_dns']       = len(net.get('dns', []))
    features['dns_requests']    = ";".join([d.get('request','') for d in net.get('dns', [])])

    # 7) Static imports
    static = report.get('static', {})
    imports = []
    for dll in static.get('pe_imports', []):
        for imp in dll.get('imports', []):
            name = imp.get('name')
            if name:
                imports.append(name)
    features['count_static_imports'] = len(imports)
    features['static_imports'] = ";".join(imports)

    # 8) Processes detail
    procs = report.get('behavior', {}).get('processes', [])
    features['count_processes'] = len(procs)
    proc_list = []
    for p in procs:
        name = p.get('process_name','')
        path = p.get('process_path','')
        pid  = p.get('pid','')
        cmd  = p.get('command_line','')
        mods = [m.get('basename') for m in p.get('modules', []) if m.get('basename')]
        proc_list.append(f"{name}|{path}|{pid}|{cmd}|{'/'.join(mods)}")
    features['processes'] = ";".join(proc_list)

    return features


def main():
    json_files = glob.glob('cuckoo_reports/report_*.json')
    all_data = []
    for file in json_files:
        with open(file,'r',encoding='utf-8') as f:
            report = json.load(f)
        feats = extract_features(report)
        feats['source_file'] = os.path.basename(file)
        all_data.append(feats)

    if not all_data:
        print('Không tìm thấy file features*.json nào.')
        return

    # Gom union fieldnames theo thứ tự xuất hiện
    fieldnames = []
    for data in all_data:
        for key in data.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open('ransomware_dataset.csv','w',newline='',encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Đã convert {len(all_data)} file features*.json vào ransomware_dataset.csv")

if __name__=='__main__':
    main()
