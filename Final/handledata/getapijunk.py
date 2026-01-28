import os
import json
from collections import Counter

def iter_api_names_from_report(report: dict):
    """
    Trả về lần lượt tên API trong 1 report.
    Hỗ trợ:
      - report đã qua extract_features (có api_call_sequence)
    """
    seq = report.get("api_call_sequence")
    if isinstance(seq, list) and seq and isinstance(seq[0], dict) and "api" in seq[0]:
        for call in seq:
            name = call.get("api")
            if name:
                yield name
        return


def build_junk_apis_from_benign(benign_dir: str, top_k: int = 200,
                                out_path: str = "junk_apis_top200_counter.json"):
    counter = Counter()

    for root, _, files in os.walk(benign_dir):
        for fname in files:
            print(fname)
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    rep = json.load(f)
                for api_name in iter_api_names_from_report(rep):
                    # with open("debug_api_names.txt", "a", encoding="utf-8") as df:
                    #     df.write(f"{api_name}\n")
                    # print(api_name)
                    counter[api_name] += 1
            except Exception as e:
                print(f"[!] Lỗi đọc {path}: {e}")

    most_common = counter.most_common(top_k)
    print(f"Top {top_k} API benign phổ biến nhất:")
    result = []
    for i, (name, cnt) in enumerate(most_common, 1):
        print(f"{i:2d}. {name} -> num calls={counter[name]}")
        result.append({"api": name, "count": cnt})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Đã lưu list vào {out_path}")


if __name__ == "__main__":
    BENIGN_DIR = "data_benign_extracted\\benign"
    build_junk_apis_from_benign(BENIGN_DIR, top_k=200,
                                 out_path="junk_apis_top200_counter.json")
