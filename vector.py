from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Load nội dung malware từ file txt
# loader = TextLoader("data/mitre_malware.txt", encoding="utf-8")
loader = TextLoader("data/technique_use_chunks.txt", encoding="utf-8")
documents = loader.load()

# Tách nội dung thành chunks (nếu cần)
# tach thanh 500 ky tu va nhung chunk sau chong len 50 ky tu tiep theo de giu lai context
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Tạo embedding model 384 chieu
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Tạo vectorstore Chroma và lưu
db = Chroma.from_documents(docs, embedding, persist_directory="vectordb5/malware5")
db.persist()

print("Vector DB đã tạo thành công.")
