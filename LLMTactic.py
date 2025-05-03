import json
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "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"


# === 1) Load feature ===
with open("features2.json", "r", encoding="utf-8") as f:
    features = json.load(f)

# === 2) Init LLM + Retriever ===
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="vectordb_mitre_attack_tatic2",
    embedding_function=embedding
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === 3) Flatten feature ===
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

# === 4) Query context + tactic ===
def query_context(feat_type: str, feat_val: str):
    if "api_call_sequence" in feat_type and feat_val.strip().startswith("{"):
        try:
            api_obj = json.loads(feat_val)
            api_name = api_obj.get("api", "")
            category = api_obj.get("category", "")
            arguments = api_obj.get("arguments", {})
            key_args = [f"{k}: {v}" for k, v in arguments.items()
                        if k in ["filename", "filepath", "regkey", "command", "host", "ip_address"]]
            arg_text = ", ".join(key_args) if key_args else json.dumps(arguments, ensure_ascii=False)
            feat_val_norm = f"{api_name} (Category: {category}, Args: {arg_text})"
        except Exception:
            feat_val_norm = feat_val
    else:
        feat_val_norm = feat_val.strip()

    # Prompt
    prompt = (
        f"You are a cybersecurity threat analyst. You are provided with behavioral features observed from sandbox execution of a suspicious binary.\n\n"
        f"Observed feature:\n- Type: {feat_type}\n- Value: {feat_val_norm}\n\n"
        f"Your task:\n"
        f"1. Identify **the single most relevant MITRE ATT&CK technique** that best matches this behavior.\n"
        f"2. Only base your answer on the retrieved MITRE context. Do not make assumptions beyond it.\n"
        f"3. If no technique clearly matches, respond exactly: \"No relevant information available.\"\n"
        f"4. Your response must strictly follow this format:\n"
        f"   - **[TXXXX – Technique Name]**: short description\n"
        f"   - Example malware usage: <if mentioned in the context>\n\n"
        f"Respond using only the specified format. Do not add extra explanations or comments."
    )

    input_key = list(rag_chain.input_keys)[0]
    output = rag_chain.invoke({input_key: prompt})
    context = output["result"].strip()
    source_docs = output["source_documents"]
    sources = [f"{doc.metadata.get('external_id')} – {doc.metadata.get('name')}" for doc in source_docs]

    # Extract tactic(s) from metadata
    tactics = set()
    for doc in source_docs:
        tactic = doc.metadata.get("tactic", "")
        for t in tactic.split(";"):
            t = t.strip()
            if t:
                tactics.add(t)

    return context, sources, sorted(tactics)

# === 5) Query all features ===
results = []
for feat_type, feat_val in flat_feats:
    context, sources, tactics = query_context(feat_type, feat_val)
    results.append(f"Feature: {feat_type} = {feat_val}")
    results.append("→ Context:\n" + context)
    results.append("→ Sources: " + (', '.join(sources) if sources else "Không có"))
    results.append("→ Tactic(s): " + (', '.join(tactics) if tactics else "Unknown"))
    results.append("\n")

# === 6) Save output ===
with open("feature_context_all_results11.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("✅ Done: Context + tactic enriched → feature_context_all_results11.txt")