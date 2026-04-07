import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
PDF_PATH = "../Gate Data Science And AI.pdf"
INDEX_PATH = "faiss_index"

def ingest_pdf():
    # 1. Load PDF
    print(f"📄 Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Split Text
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Vector Store
    print("🧠 Generating embeddings and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # We use LangChain's FAISS wrapper which handles chunk storage automatically
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Save locally
    print(f"💾 Saving vector store to {INDEX_PATH}...")
    vector_store.save_local(INDEX_PATH)
    
    print("✅ Ingestion complete (LangChain version)!")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
    else:
        ingest_pdf()
