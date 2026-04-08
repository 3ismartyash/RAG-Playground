import pickle
import os

# Path to the pickle file
PKL_PATH = os.path.join("no_framework", "chunks.pkl")

if os.path.exists(PKL_PATH):
    with open(PKL_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks)} chunks from {PKL_PATH}")
    print("-" * 50)

    # Show the first 2 chunks fully
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i} ---")
        print(chunk) 
        print("\n")
else:
    print(f"❌ File not found at {PKL_PATH}")
