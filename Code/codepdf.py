# # 📦 Pipeline: Ingest CTI từ PDF (unstructured) vào vector DB (Chroma)

# import os
# import fitz  # PyMuPDF
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document

# # === Config ===
# PDF_FOLDER = "data"
# VECTOR_DIR = "vectordbpdf"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
# all_docs = []

# # === 1. Extract PDF text ===
# def extract_text_from_pdf(path):
#     try:
#         doc = fitz.open(path)
#         text = "\n".join([page.get_text() for page in doc])
#         return text
#     except Exception as e:
#         print(f"[!] Lỗi đọc PDF {path}: {e}")
#         return ""

# # === 2. Ingest PDF CTI (blog, báo cáo) ===
# def load_pdf_to_documents(folder):
#     docs = []
#     for fname in os.listdir(folder):
#         if fname.endswith(".pdf"):
#             fpath = os.path.join(folder, fname)
#             raw_text = extract_text_from_pdf(fpath)
#             if raw_text.strip():
#                 chunks = text_splitter.split_text(raw_text)
#                 for chunk in chunks:
#                     docs.append(Document(page_content=chunk, metadata={"source": fname, "type": "pdf"}))
#     return docs

# # === Run pipeline ===
# print("🔄 Đang nạp CTI từ PDF...")
# all_docs.extend(load_pdf_to_documents(PDF_FOLDER))
# # print(all_docs)
# print(f"📄 Tổng số document: {len(all_docs)}")

# print("💾 Đang tạo vector database...")
# db = Chroma.from_documents(all_docs, embedding, persist_directory=VECTOR_DIR)
# db.persist()
# print("✅ Vector database đã được tạo ở:", VECTOR_DIR)












# # 📦 Pipeline: Ingest CTI từ PDF (unstructured) vào vector DB (Chroma) kèm OCR ảnh minh họa

# import os
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import cv2
# import numpy as np
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document

# # === Config ===
# PDF_FOLDER = "data"
# VECTOR_DIR = "vectordbpdf"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
# IMG_DIR = "pdf_images"
# os.makedirs(IMG_DIR, exist_ok=True)

# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
# all_docs = []

# def preprocess_for_ocr(pil_image):
#     img = np.array(pil_image)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
#     resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     return Image.fromarray(resized)

# def extract_text_from_pdf(path):
#     try:
#         doc = fitz.open(path)
#         all_text = []
#         for page_num, page in enumerate(doc):
#             page_text = page.get_text()
#             all_text.append(page_text)

#             images = page.get_images(full=True)
#             for img_index, img in enumerate(images):
#                 xref = img[0]
#                 base_img = doc.extract_image(xref)
#                 img_bytes = base_img["image"]
#                 img_ext = base_img["ext"]
#                 img_name = f"page_{page_num+1}_img_{img_index}.{img_ext}"
#                 img_path = os.path.join(IMG_DIR, img_name)
#                 with open(img_path, "wb") as f:
#                     f.write(img_bytes)

#                 try:
#                     img_pil = Image.open(img_path).convert("RGB")
#                     preprocessed = preprocess_for_ocr(img_pil)
#                     ocr_text = pytesseract.image_to_string(preprocessed, lang="eng")
#                     if ocr_text.strip():
#                         all_text.append("[IMAGE OCR]\n" + ocr_text.strip())
#                 except Exception as e:
#                     print(f"[!] OCR lỗi ở trang {page_num+1}, ảnh {img_index}: {e}")

#         return "\n".join(all_text)
#     except Exception as e:
#         print(f"[!] Lỗi đọc PDF {path}: {e}")
#         return ""

# def load_pdf_to_documents(folder):
#     docs = []
#     for fname in os.listdir(folder):
#         if fname.endswith(".pdf"):
#             fpath = os.path.join(folder, fname)
#             raw_text = extract_text_from_pdf(fpath)
#             if raw_text.strip():
#                 chunks = text_splitter.split_text(raw_text)
#                 for chunk in chunks:
#                     docs.append(Document(page_content=chunk, metadata={"source": fname, "type": "pdf+ocr"}))
#     return docs

# # === Run pipeline ===
# print("🔄 Đang nạp CTI từ PDF...")
# all_docs.extend(load_pdf_to_documents(PDF_FOLDER))
# print(f"📄 Tổng số document: {len(all_docs)}")

# print("💾 Đang tạo vector database...")
# db = Chroma.from_documents(all_docs, embedding, persist_directory=VECTOR_DIR)
# db.persist()
# print("✅ Vector database đã được tạo ở:", VECTOR_DIR)





# 📦 Pipeline: Ingest cả CTI STIX + PDF (text only) vào 1 vector DB chung (Chroma)

import os
import fitz  # PyMuPDF
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === Config ===
STIX_FILE = "data/technique_to_malware.txt"
PDF_FOLDER = "data"
VECTOR_DIR = "vectordb_all"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_docs = []

# === 1. Load STIX documents (from txt)
def load_stix_to_documents(stix_file):
    with open(stix_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source_type": "stix"}) for chunk in chunks]

# === 2. Load PDF (text only)
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f"[!] Lỗi đọc PDF {path}: {e}")
        return ""

def load_pdf_to_documents(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            fpath = os.path.join(folder, fname)
            raw_text = extract_text_from_pdf(fpath)
            if raw_text.strip():
                chunks = text_splitter.split_text(raw_text)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata={"source_type": "pdf", "source": fname}))
    return docs

# === 3. Run pipeline ===
print("📥 Đang load CTI từ STIX...")
docs_stix = load_stix_to_documents(STIX_FILE)
print(f"✅ {len(docs_stix)} đoạn STIX")

print("📥 Đang load CTI từ PDF...")
docs_pdf = load_pdf_to_documents(PDF_FOLDER)
print(f"✅ {len(docs_pdf)} đoạn PDF")

all_docs = docs_stix + docs_pdf
print(f"📦 Tổng cộng: {len(all_docs)} documents")

print("💾 Đang tạo vector database...")
db = Chroma.from_documents(all_docs, embedding, persist_directory=VECTOR_DIR)
db.persist()
print("✅ Vector database đã được tạo tại:", VECTOR_DIR)


