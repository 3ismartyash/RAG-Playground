import os
import faiss
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SETTINGS ---
# You can change this to any of your index paths
INDEX_PATH = os.path.join("langGraph", "faiss_index")
#INDEX_PATH = os.path.join("langchain", "faiss_index")
#INDEX_PATH = os.path.join("no_framework", "vector_index.faiss")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def inspect_raw_vectors():
    print(f"🚀 Initializing Raw Vector Inspector for: {INDEX_PATH}")
    print("-" * 50)

    # 1. Load the Index
    if INDEX_PATH.endswith(".faiss"):
        # Raw FAISS file
        index = faiss.read_index(INDEX_PATH)
    else:
        # LangChain folder
        abs_path = os.path.abspath(INDEX_PATH)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(abs_path, embeddings, allow_dangerous_deserialization=True)
        index = vector_store.index

    # 2. Extract Technical Details
    print(f"📊 DATA STRUCTURE TYPE: {type(index)}")
    print(f"📐 DIMENSIONS (Coordinates per vector): {index.d}")
    print(f"🔢 TOTAL VECTORS STORED: {index.ntotal}")
    print("-" * 50)

    # 3. Pull the First Vector (Chunk 0)
    # reconstruct(0) retrieves the actual float32 array stored in memory
    raw_vector = index.reconstruct(0)

    print("\n💎 RAW VECTOR DATA (Chunk 0):")
    print(f"• Numpy Shape: {raw_vector.shape}")
    print(f"• Memory Size: {raw_vector.nbytes} bytes")
    print("\n--- FIRST 20 DIMENSIONS ---")
    print(raw_vector[:20])
    print("\n--- LAST 20 DIMENSIONS ---")
    print(raw_vector[-20:])

    print("\n" + "="*50)
    print("💡 NOTE: These numbers represent the 'semantic location' of your text.")
    print("In FAISS, this is stored as a contiguous C++ array (matrix).")

if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        # Try one level up if not found (since we are in 'diagnostics' folder)
        INDEX_PATH = os.path.join("..", INDEX_PATH.replace("..\\", ""))
        
    if os.path.exists(INDEX_PATH):
        inspect_raw_vectors()
    else:
        print(f"❌ Error: Index not found at {INDEX_PATH}")
