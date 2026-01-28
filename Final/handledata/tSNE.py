import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# =========================
# CONFIG
# =========================
DIR_2021  = r"ransomware"
EARLY_DIR = r"early_clean_2_10"
LATE_DIR  = r"late_clean_2_10"

MAX_LEN = 1500
NGRAM_RANGE = (2, 4)
SEED = 42

# =========================
# LOAD (KEEP ALL)
# =========================
def load_folder(dir_path, group):
    rows = []
    for fp in glob.glob(os.path.join(dir_path, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)

            seq = j.get("api_call_sequence", None)

            # 1) Không có / sai kiểu -> gom chung NO_API (lặp 2 lần để có bigram)
            if not isinstance(seq, list):
                seq = ["<NO_API>", "<NO_API>"]
            else:
                seq = [str(x).strip().lower() for x in seq if str(x).strip()]

                # 2) Rỗng -> NO_API
                if len(seq) == 0:
                    seq = ["<NO_API>", "<NO_API>"]

                # 3) Chỉ có 1 API -> thêm EOS để tạo bigram
                elif len(seq) == 1:
                    seq = [seq[0], "<EOS>"]

            # 4) Cắt độ dài (để tránh quá nặng)
            if len(seq) > MAX_LEN:
                seq = seq[:MAX_LEN]

            rows.append({"path": fp, "group": group, "seq": seq})

        except Exception:
            # nếu muốn debug: print("[LOAD_FAIL]", fp)
            continue
    return rows

df = pd.DataFrame(
    load_folder(DIR_2021,  "y2021") +
    load_folder(EARLY_DIR, "early_2025") +
    load_folder(LATE_DIR,  "late_2025")
)

print("Loaded:", df.shape)
print(df["group"].value_counts())

no_api_count = df["seq"].apply(lambda s: s == ["<NO_API>", "<NO_API>"]).sum()
print(f"NO_API samples: {no_api_count} / {len(df)}")

# =========================
# VECTORIZE: TF-IDF on n-grams
# =========================
texts = [" ".join(seq) for seq in df["seq"].tolist()]

vec = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[^ ]+",
    ngram_range=NGRAM_RANGE,
    min_df=2,
    max_df=0.95
)
X = vec.fit_transform(texts)

# giảm chiều trước khi t-SNE để nhanh + ổn định
svd = TruncatedSVD(n_components=50, random_state=SEED)
X50 = svd.fit_transform(X)

# =========================
# t-SNE
# =========================
perp = 30
perp = min(perp, max(5, (len(df) - 1) // 3))

tsne = TSNE(
    n_components=2,
    perplexity=perp,
    init="pca",
    learning_rate="auto",
    random_state=SEED
)
Z = tsne.fit_transform(X50)

# =========================
# PLOT
# =========================
plt.figure(figsize=(9, 7))
for g in ["y2021", "early_2025", "late_2025"]:
    idx = (df["group"].values == g)
    plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.7, label=g)

plt.title(f"t-SNE scatter 2021 vs clean 2-10 (TF-IDF API n-grams {NGRAM_RANGE})")
plt.legend()
plt.tight_layout()
plt.show()
