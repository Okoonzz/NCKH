# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA

# # Thiết lập OpenAI key
# os.environ["OPENAI_API_KEY"] = "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"  # <-- THAY bằng key của bạn

# # Load vector store
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="vectordb/malware", embedding_function=embedding)

# # Load LLM từ OpenAI
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Tạo pipeline RAG
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
#     return_source_documents=True
# )

# # Prompt mẫu
# query = input(" Nhập câu hỏi về malware: ")
# result = qa(query)

# # Hiển thị kết quả
# print("\n Kết quả trả về:\n")
# print(result['result'])

# print("\n Nguồn tham chiếu:")
# for i, doc in enumerate(result['source_documents']):
#     print(f"- Đoạn {i+1}:\n{doc.page_content}\n")





import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Thiết lập OpenAI Key
os.environ["OPENAI_API_KEY"] = "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"

# # Load embedding model
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Load các vector DB
# malware_tech_db = Chroma(persist_directory="vectordb2/malware2", embedding_function=embedding)
# tech_to_malware_db = Chroma(persist_directory="vectordb3/malware3", embedding_function=embedding)
# # malware_use_db = Chroma(persist_directory="vectordb4/malware4", embedding_function=embedding)
# # malware_use_pdf = Chroma(persist_directory="vectordbpdf", embedding_function=embedding)
# # malware_use_all = Chroma(persist_directory="vectordb_all", embedding_function=embedding)

# # Tạo retrievers
# malware_tech_retriever = malware_tech_db.as_retriever(search_kwargs={"k": 10})
# tech_to_malware_retriever = tech_to_malware_db.as_retriever(search_kwargs={"k": 20})
# # malware_use_retriever = malware_use_db.as_retriever(search_kwargs={"k": 10})
# # malware_use_retriever_pdf = malware_use_pdf.as_retriever(search_kwargs={"k": 10})
# # malware_use_retriever_all = malware_use_all.as_retriever(search_kwargs={"k": 10})

# # Tạo LLM từ OpenAI
# # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.2)

# # Tạo các RetrievalQA chains
# malware_tech_chain = RetrievalQA.from_chain_type(llm=llm, retriever=malware_tech_retriever, return_source_documents=True)
# tech_to_malware_chain = RetrievalQA.from_chain_type(llm=llm, retriever=tech_to_malware_retriever, return_source_documents=True)
# # malware_use_chain = RetrievalQA.from_chain_type(llm=llm, retriever=malware_use_retriever, return_source_documents=True)
# # malware_use_pdff = RetrievalQA.from_chain_type(llm=llm, retriever=malware_use_retriever_pdf, return_source_documents=True)
# # malware_use_all_chain = RetrievalQA.from_chain_type(llm=llm, retriever=malware_use_retriever_all, return_source_documents=True)

# # Lấy câu hỏi từ người dùng
# query = input(" Nhập câu hỏi: ").strip().lower()

# # Chọn hướng truy vấn dựa vào nội dung câu hỏi
# try:
#     if "mô tả" in query or "hành vi" in query or "use" in query or "beacon" in query:
#         print("\n--------- Truy vấn theo chiều: USE (mô tả hành vi) ➝ Malware")
#         # input_key = list(malware_use_chain.input_keys)[0]
#         # rag_result = malware_use_chain.invoke({input_key: query})

#     elif "kỹ thuật" in query and ("malware" in query or "phần mềm" in query):
#         print("\n------------- Truy vấn theo chiều: Kỹ thuật ➝ Malware")
#         input_key = list(tech_to_malware_chain.input_keys)[0]
#         rag_result = tech_to_malware_chain.invoke({input_key: query})

#     else:
#         print("\n------------ Truy vấn theo chiều: Malware ➝ Kỹ thuật")
#         input_key = list(malware_tech_chain.input_keys)[0]
#         rag_result = malware_tech_chain.invoke({input_key: query})

#     context = rag_result["result"]
#     sources = rag_result.get("source_documents", [])
#     print(context)

#     # Nếu không có thông tin rõ ràng từ CTI ➝ fallback GPT
#     # if ("không có thông tin" in context.lower()) or len(context.strip()) < 30:
#     #     print("\n<=============> CTI nội bộ không đủ thông tin. Đang truy vấn GPT toàn cục...")
#     #     fallback_response = llm.predict(query)
#     #     print("\n<=============> Kết quả từ GPT:")
#     #     print(fallback_response)
#     # else:
#     #     print("\n<=============> Kết quả từ CTI nội bộ:")
#     #     print(context)

# except Exception as e:
#     print(" Lỗi khi truy vấn:", e)



# vectordb_all = Chroma(persist_directory="vectordb_all", embedding_function=embedding)

# # Tạo 2 retrievers có filter metadata
# retriever_stix = vectordb_all.as_retriever(search_kwargs={"k": 150, "filter": {"source_type": "stix"}})
# retriever_pdf = vectordb_all.as_retriever(search_kwargs={"k": 150, "filter": {"source_type": "pdf"}})

# # Tạo LLM từ OpenAI
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# # Tạo RetrievalQA chain riêng cho STIX và PDF
# qa_stix = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_stix, return_source_documents=True)
# qa_pdf = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_pdf, return_source_documents=True)

# # Lấy câu hỏi từ người dùng
# query = input("\U0001F9E0 Nhập câu hỏi: ").strip().lower()

# # Truy vấn STIX trước, fallback PDF nếu cần
# try:
#     input_key = list(qa_stix.input_keys)[0]
#     rag_result = qa_stix.invoke({input_key: query})

#     context = rag_result["result"]
#     sources = rag_result.get("source_documents", [])

#     if ("không có thông tin" in context.lower()) or len(context.strip()) < 30:
#         print("\n<=============> STIX không đủ thông tin. Fallback sang PDF...")
#         input_key = list(qa_pdf.input_keys)[0]
#         rag_result = qa_pdf.invoke({input_key: query})
#         context = rag_result["result"]

#     print("\n<=============> KếT QUẢ:")
#     print(context)

# except Exception as e:
#     print("\n❌ Lỗi khi truy vấn:", e)



##################################################################################################################

# # 1) Khởi tạo embedding + LLM
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# # 2) Load lại vector DB vừa build (kỹ thuật + examples)
# mitre_db = Chroma(
#     persist_directory="vectordb_mitre_attack",
#     embedding_function=embedding
# )
# mitre_retriever = mitre_db.as_retriever(search_kwargs={"k": 2})

# # 3) Tạo RetrievalQA chain
# mitre_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=mitre_retriever,
#     return_source_documents=True
# )

# # 4) Hàm query context cho một feature
# def query_context(feat_type: str, feat_val: str):
#     prompt = (
#         f"Feature hành vi: **{feat_type}** = `{feat_val}`\n"
#         "Theo MITRE ATT&CK, hành vi này thường liên quan đến kỹ thuật nào? "
#         "Trả lời ở dạng:\n"
#         "- **[TXXXX – Technique Name]**: mô tả ngắn\n"
#         "- Ví dụ malware sử dụng: <liệt kê nếu có>\n"
#     )
#     input_key = list(mitre_chain.input_keys)[0]
#     rag_out   = mitre_chain.invoke({input_key: prompt})

#     # Kết quả text
#     context = rag_out["result"]

#     # Lấy metadata của các tài liệu nguồn (external_id + name)
#     sources = [
#         f"{d.metadata.get('external_id')} – {d.metadata.get('name')}"
#         for d in rag_out["source_documents"]
#     ]
#     return context, sources

# # 5) Thử với một feature khác: regkey_opened
# feat_type  = "regkey_opened"
# feat_value = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\OLE"

# context, sources = query_context(feat_type, feat_value)
# print(f"\n🔹 Feature: {feat_type} = {feat_value}")
# print("→ Kỹ thuật + mô tả:\n", context)
# print("→ Nguồn (external_id – name):", sources)

####################################################################################################################


# 1) Load your feature.json
with open("features.json", "r", encoding="utf-8") as f:
    features = json.load(f)

# 2) Initialize embedding, vector DB, retriever, and LLM
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="vectordb_mitre_attack",
    embedding_function=embedding
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 3) Helper to flatten nested JSON features into (type, value) pairs
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
            if isinstance(val, (dict, list)):
                val_str = json.dumps(val, ensure_ascii=False)
            else:
                val_str = str(val)
            items.append((feat_type, val_str))
    else:
        feat_type = "_".join(prefix)
        items.append((feat_type, str(obj)))
    return items

flat_feats = flatten_features(features)

# 4) Query function for a single feature
def query_context(feat_type: str, feat_val: str):
    prompt = (
        f"Feature hành vi: **{feat_type}** = `{feat_val}`\n"
        "Theo MITRE ATT&CK, hành vi này liên quan đến tối đa 2 kỹ thuật chính. "
        "Trả lời ở dạng:\n"
        "- **[TXXXX – Technique Name]**: mô tả ngắn\n"
        "- Ví dụ malware sử dụng: <nếu có>\n"
    )
    input_key = list(rag_chain.input_keys)[0]
    output = rag_chain.invoke({input_key: prompt})
    context = output["result"].strip()
    sources = [
        f"{doc.metadata.get('external_id')} – {doc.metadata.get('name')}"
        for doc in output["source_documents"]
    ]
    return context, sources

# 5) Iterate all features, query and collect results
results = []
for feat_type, feat_val in flat_feats:
    context, sources = query_context(feat_type, feat_val)
    results.append(f"Feature: {feat_type} = {feat_val}")
    results.append("→ Kỹ thuật + mô tả:\n" + context)
    results.append("→ Nguồn: " + (", ".join(sources) if sources else "Không có"))
    results.append("\n")

# 6) Write all results to a text file
with open("feature_context_all_results2.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(results))

print("✅ Saved all feature contexts to 'feature_context_all_results2.txt'")

######################################################################################################3

