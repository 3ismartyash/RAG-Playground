import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
INDEX_PATH = "vector_index.faiss"
CHUNKS_PATH = "chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def load_index():
    print(f"Loading index and chunks...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def get_context(query, index, chunks, top_k=3):
    print(f"Searching for most relevant chunks...")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return "\n---\n".join(retrieved_chunks)

def generate_answer(query, context):
    print(f"Generating answer using Google Gemini...")
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    Context from the PDF:
    {context}
    
    Question:
    {query}
    
    Please provide an accurate answer based ONLY on the context provided above.
    """
    
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("Error: Index or chunks not found. Please run ingest.py first.")
    else:
        index, chunks = load_index()
        
        while True:
            user_query = input("\n Ask anything about the PDF (or write 'exit'): ")
            if user_query.lower() == 'exit':
                break

            # Step 1: Semantic Search
            context = get_context(user_query, index, chunks)
            
            # Step 2: Generation
            try:
                answer = generate_answer(user_query, context)
                print(f"\n Answer: \n{answer}")
            except Exception as e:
                print(f"Error while calling Gemini: {e}")
