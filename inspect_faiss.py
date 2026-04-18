import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SETTINGS ---
# Uncomment ONLY ONE of these three:
#INDEX_PATH = os.path.join("langExtract", "faiss_index") 
INDEX_PATH = os.path.join("langGraph", "faiss_index") 
#INDEX_PATH = os.path.join("langchain", "faiss_index")
#INDEX_PATH = os.path.join("no_framework", "vector_index.faiss")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print(f"🔍 Inspecting: {INDEX_PATH}")

# --- INSPECTION LOGIC ---

# 1. Check if it's the "No-Framework" raw file
if INDEX_PATH.endswith(".faiss"):
    print("🛠️ Detected Raw FAISS file (No-Framework version)")
    CHUNKS_PATH = os.path.join("no_framework", "chunks.pkl")
    
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        # Load raw binary index
        index = faiss.read_index(INDEX_PATH)
        # Load raw text chunks
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
            
        print(f"📊 Total Vectors: {index.ntotal}")
        print(f"📦 Total Chunks: {len(chunks)}")
        
        # Show first 2
        for i in range(min(2, len(chunks))):
            print(f"\n--- Chunk {i} ---")
            print(f"📄 Content: {chunks[i][:300]}...")
    else:
        print(f"❌ Missing files in no_framework! Need both {INDEX_PATH} and {CHUNKS_PATH}")

# 2. Otherwise, treat it as a LangChain/LangGraph folder
else:
    print("🦜 Detected LangChain/LangGraph folder")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(INDEX_PATH):
        # LangChain's load_local expects a folder path
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"📊 Total Vectors: {vector_store.index.ntotal}")
        
        doc_dict = vector_store.docstore._dict
        first_two_keys = list(doc_dict.keys())[:2]
        
        for doc_id in first_two_keys:
            doc = doc_dict[doc_id]
            print(f"\n--- Document ID: {doc_id} ---")
            print(f"📄 Content: {doc.page_content[:300]}...")
            print(f"🏷️ Metadata: {doc.metadata}")
    else:
        print(f"❌ Index folder not found: {INDEX_PATH}")
