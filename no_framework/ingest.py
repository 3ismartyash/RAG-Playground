import os
from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

PDF_PATH = "../Gate Data Science And AI.pdf"
INDEX_PATH = "vector_index.faiss"
CHUNKS_PATH = "chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    print(f"Chunking text (size={chunk_size}, overlap={overlap})...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def build_vector_db(chunks):
    print(f"Generating embeddings using {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks)
    
    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving index and chunks...")
    faiss.write_index(index, INDEX_PATH)
    
    # We save chunks in a pickle file because FAISS doesn't store the metadata (text)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found.")
    else:
        raw_text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(raw_text)
        build_vector_db(chunks)
