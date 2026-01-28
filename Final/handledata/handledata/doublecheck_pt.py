import os
import torch

root = "pyg_data_DistilBERT_data_ransomware"

with open("doublecheck_labels_rans4.txt", "a", encoding="utf-8") as fout:
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.endswith(".pt"):
                path = os.path.join(subdir, f)
                data = torch.load(path, map_location="cpu", weights_only=False)

                folder = os.path.basename(subdir)
                gt = 1 if folder == "ransomware" else 0

                if hasattr(data, "y"):
                    y = int(data.y.item())
                    # ok = (y == gt)
                    fout.write(f"name: {f} | label={y} |\n")
                else:
                    print(f"name: {f} | label={y} | KHÔNG có label trong Data")
