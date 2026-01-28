import requests
import time
import os
import json
import shutil

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CUCKOO_API = "http://192.168.111.158:8090"
CUCKOO_STORAGE = "/home/cuckoo/.cuckoo/storage/analyses"
SAMPLES_DIR = "resample"
REPORT_DIR = "cuckoo_reports_resample"
REPORT_LIST = "cuckoo_reports_resamples.txt"

os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load danh sách file đã có report
# ---------------------------------------------------------
def load_completed_files():
    if not os.path.exists(REPORT_LIST):
        return set()
    with open(REPORT_LIST, "r") as f:
        return set(line.strip() for line in f.readlines())


# ---------------------------------------------------------
# Ghi thêm tên file vào report list
# ---------------------------------------------------------
def append_completed_file(filename):
    with open(REPORT_LIST, "a") as f:
        f.write(filename + "\n")


# ---------------------------------------------------------
# Xóa folder analysis
# ---------------------------------------------------------
def delete_task(task_id):
    task_dir = os.path.join(CUCKOO_STORAGE, str(task_id))
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
        print(f"[+] Deleted task {task_id} storage.")
    else:
        print(f"[!] Task {task_id} folder not found.")


# ---------------------------------------------------------
# Submit file
# ---------------------------------------------------------
def submit_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            r = requests.post(f"{CUCKOO_API}/tasks/create/file", files={'file': f})
            r.raise_for_status()
            task_id = r.json()['task_id']
            print(f"[+] Submitted {file_path} → Task {task_id}")
            return task_id
    except Exception as e:
        print(f"[!] Failed to submit {file_path}: {e}")
        return None


# ---------------------------------------------------------
# Wait for report
# ---------------------------------------------------------
def wait_for_report(task_id):
    while True:
        try:
            r = requests.get(f"{CUCKOO_API}/tasks/view/{task_id}")
            r.raise_for_status()
            status = r.json()['task']['status']
        except Exception as e:
            print(f"[!] Error checking task {task_id}: {e}")
            time.sleep(5)
            continue

        print(f"    Task {task_id} status = {status}")

        if status == "reported":
            print(f"[+] Task {task_id} completed.")
            break

        time.sleep(5)


# ---------------------------------------------------------
# Download report JSON
# ---------------------------------------------------------
def download_report(task_id, filepath, filename):
    try:
        r = requests.get(f"{CUCKOO_API}/tasks/report/{task_id}/json")
        r.raise_for_status()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(r.json(), f, indent=2)

        print(f"[+] Saved report → {filepath}")

        # Thêm tên file vào txt
        append_completed_file(filename)

    except Exception as e:
        print(f"[!] Failed to download report: {e}")


# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------
def main():
    completed = load_completed_files()
    files = sorted(os.listdir(SAMPLES_DIR))

    for fname in files:
        file_path = os.path.join(SAMPLES_DIR, fname)

        # Skip nếu đã có trong txt
        if fname in completed:
            print(f"[i] Skip {fname}, report already exists.")
            continue

        print("\n===============================")
        print(f"[+] Running: {fname}")
        print("===============================")

        # 1. Submit
        task_id = submit_file(file_path)
        if not task_id:
            continue

        # 2. Wait
        wait_for_report(task_id)

        # 3. Download report
        save_path = os.path.join(REPORT_DIR, f"report_{fname}.json")
        download_report(task_id, save_path, fname)

        # 4. Delete storage
        delete_task(task_id)

        print(f"[+] DONE {fname}")
        print("-------------------------------")

    print("\n[✓] All files processed.")


if __name__ == "__main__":
    main()

