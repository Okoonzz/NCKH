import requests
from datetime import datetime, timezone
import os
import time
import shutil

API_KEY = ""  

PATHFOLDERDATAOLD = r"chualoc"
PATHFOLDERDATANEW = r"ori_data"
PATHFOLDERDONE = r"datanew"

headers = {"x-apikey": API_KEY}


def query_virustotal(url, headers) -> int:
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json().get("data", {}).get("attributes", {})

        def ts(t):
            return datetime.fromtimestamp(t, tz=timezone.utc).isoformat() if t else None

        sha256 = data.get("sha256")
        first_sub = data.get("first_submission_date")
        file_type = (data.get("type_description") or "").lower()

        year = None
        if first_sub:
            year = datetime.fromtimestamp(first_sub, tz=timezone.utc).year

        if year and year == 2021 and ("exe" in file_type):
            print(f"Lấy file: {sha256}")
            print("First submission:", ts(first_sub))
            print("File type:", file_type)

            with open("vt12_results.txt", "a", encoding="utf-8") as f:
                f.write(f"SHA256: {sha256}\n")
                f.write(f"First submission: {ts(first_sub)}\n")
                f.write("\n")
            return 1
        else:
            print(f"file: {sha256} (year: {year}, type: {file_type})")
            return 0

    else:
        print("Lỗi:", r.status_code, r.text[:400])
    return 0


def findhashnew(pathfolder=PATHFOLDERDATAOLD, outfolder=PATHFOLDERDATANEW):
    limitapi = 0

    for fname in os.listdir(pathfolder):

        if limitapi >= 500:
            print("Reached API limit for this run.")
            break

        gethash = fname.split("report_")[-1].rsplit(".json", 1)[0]
        if '.' in gethash: 
            gethash = gethash.split('.', 1)[0]

        url = f"https://www.virustotal.com/api/v3/files/{gethash}"
        print(f"\nQuerying hash: {gethash}")
        res = query_virustotal(url, headers)
        limitapi += 1


        if res == 1:
            src_path = os.path.join(pathfolder, fname)
            dst_path = os.path.join(outfolder, fname)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {fname} → {outfolder}")
            time.sleep(20)

if __name__ == "__main__":
    findhashnew()