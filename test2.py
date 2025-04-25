import json
import os
import re
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "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"

# === Step 1: Load features.json ===
with open("features2.json", "r", encoding="utf-8") as f:
    features = json.load(f)

# === Step 2: Flatten features ===
def flatten_features(obj, prefix=None):
    if prefix is None:
        prefix = []
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.extend(flatten_features(v, prefix + [k]))
    elif isinstance(obj, list):
        feat_type = "_".join(prefix)
        for val in obj:
            val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
            items.append((feat_type, val_str))
    else:
        feat_type = "_".join(prefix)
        items.append((feat_type, str(obj)))
    return items

flat_feats = flatten_features(features)

# === Step 3: Normalize value for better search ===
def normalize_feature_value(feat_type: str, feat_val: str) -> str:
    val = feat_val.lower()
    if "regkey" in feat_type:
        val = re.sub(r"^hkey_local_machine\\\\?", "HKLM\\\\", val, flags=re.IGNORECASE)
    if "dll" in feat_type:
        val = val.split("\\")[-1]
    if "file" in feat_type and val.endswith(".exe"):
        val = "execution of file: " + val.split("\\")[-1]
    return val

# === Step 4: Load vector DB + RAG chain ===
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="vectordb_mitre_attack",
    embedding_function=embedding
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Step 5: Prompt and query context ===
def query_context(feat_type: str, feat_val: str):
    feat_val_norm = normalize_feature_value(feat_type, feat_val)
    prompt = (
        f"Tôi đang phân tích hành vi phần mềm độc hại. Dưới đây là một đặc trưng hành vi:\n\n"
        f"- Loại đặc trưng: **{feat_type}**\n"
        f"- Giá trị cụ thể: `{feat_val}`\n"
        f"- Sau khi chuẩn hóa, tôi có thể hiểu là: `{feat_val_norm}`\n\n"
        f"Đặc trưng này tương ứng với kỹ thuật nào trong MITRE ATT&CK? "
        f"Cho tôi tối đa 2 kỹ thuật phù hợp, bao gồm:\n"
        "- **[Txxxx – Tên kỹ thuật]**: mô tả ngắn (từ dữ liệu MITRE)\n"
        "- Ví dụ phần mềm độc hại đã từng sử dụng kỹ thuật này (nếu có)\n"
    )
    input_key = list(rag_chain.input_keys)[0]
    output = rag_chain.invoke({input_key: prompt})
    context = output["result"].strip()
    sources = [
        f"{doc.metadata.get('external_id')} – {doc.metadata.get('name')}"
        for doc in output["source_documents"]
    ]
    return context, sources

# === Step 6: Query all features and store ===
results = []
for feat_type, feat_val in flat_feats:
    context, sources = query_context(feat_type, feat_val)
    results.append(f"Feature: {feat_type} = {feat_val}")
    results.append("→ Kỹ thuật + mô tả:\n" + context)
    results.append("→ Nguồn: " + (", ".join(sources) if sources else "Không có"))
    results.append("\n")

# === Step 7: Save results ===
with open("feature_context_all_results2.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(results))

print("✅ Đã lưu kết quả vào feature_context_all_results2.txt")
