
import json
import matplotlib.pyplot as plt

# Load file
with open("junk_apis_ordered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy top 20
top_k = 25
top20 = data[:]

apis = [item["api"] for item in top20]
counts = [item["count"] for item in top20]

# Vẽ biểu đồ
plt.figure()
plt.bar(apis, counts)
plt.xticks(rotation=75, ha='right')
plt.title(f"Top {top_k} Junk APIs by Frequency")
plt.xlabel("API Name")
plt.ylabel("Count")

plt.ticklabel_format(style='plain', axis='y')
plt.gca().get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)

plt.tight_layout()
plt.show()