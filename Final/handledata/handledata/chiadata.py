import csv
import shutil
import os

CSV_FILE = "metadata_KB1_Q.csv"          
SOURCE_DIR_NOISE = "data_ransomware2025_extracted_noise\\ransomware"
SOURCE_DIR_CLEAN = "data_ransomware_extracted2025\\ransomware"


def createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR, CSV_FILE, label):
    os.makedirs(EARLY_DIR, exist_ok=True)
    os.makedirs(LATE_DIR, exist_ok=True)

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            filename = row["filename"].strip()
            label_q1 = row[f"{label}"].strip()
            domain = row["domain"].strip()

            src_path = os.path.join(SOURCE_DIR, filename)

            if not os.path.isfile(src_path):
                # print(f"[!] File không tồn tại: {src_path}")
                continue
            if domain == 'target' and label_q1 == 'train':
                dst_path = os.path.join(EARLY_DIR, filename)
                # print(f"[+] Copied {filename} -> {EARLY_DIR}")
            elif domain == 'target' and label_q1 == 'test':
                dst_path = os.path.join(LATE_DIR, filename)
                # print(f"[+] Copied {filename} -> {LATE_DIR}")
            else:
                continue  # bỏ qua các giá trị khác

            shutil.copy2(src_path, dst_path)
            print(f"[+] Copied {filename} -> {dst_path}")
    
    print("===========done write===========")

def noise48(SOURCE_DIR_NOISE, CSV_FILE):
    EARLY_DIR = "early_noise_4_8"
    LATE_DIR = "late_noise_4_8"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_NOISE, CSV_FILE, "label_Q1")
    print("===========done noise 4-8===========")
    return


def noise66(SOURCE_DIR_NOISE, CSV_FILE):
    EARLY_DIR = "early_noise_6_6"
    LATE_DIR = "late_noise_6_6"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_NOISE, CSV_FILE, "label_Q2")
    print("===========done noise 6-6===========")
    return

def noise28(SOURCE_DIR_NOISE, CSV_FILE):
    EARLY_DIR = "early_noise_2_10"
    LATE_DIR = "late_noise_2_10"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_NOISE, CSV_FILE, "split_KB1")
    print("===========done noise 2-10===========")
    return

def clean48(SOURCE_DIR_CLEAN, CSV_FILE):
    EARLY_DIR = "early_clean_4_8"
    LATE_DIR = "late_clean_4_8"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_CLEAN, CSV_FILE, "label_Q1")
    print("===========done clean 4-8===========")
    return

def clean66(SOURCE_DIR_CLEAN, CSV_FILE):
    EARLY_DIR = "early_clean_6_6"
    LATE_DIR = "late_clean_6_6"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_CLEAN, CSV_FILE, "label_Q2")
    print("===========done clean 6-6===========")
    return

def clean28(SOURCE_DIR_CLEAN, CSV_FILE):
    EARLY_DIR = "early_clean_2_10"
    LATE_DIR = "late_clean_2_10"
    createfile(EARLY_DIR, LATE_DIR, SOURCE_DIR_CLEAN, CSV_FILE, "split_KB1")
    print("===========done clean 2-10===========")
    return

def main():
    noise48(SOURCE_DIR_NOISE, CSV_FILE)
    noise66(SOURCE_DIR_NOISE, CSV_FILE)
    noise28(SOURCE_DIR_NOISE, CSV_FILE)
    clean48(SOURCE_DIR_CLEAN, CSV_FILE)
    clean66(SOURCE_DIR_CLEAN, CSV_FILE)
    clean28(SOURCE_DIR_CLEAN, CSV_FILE)

if __name__ == "__main__":
    main()