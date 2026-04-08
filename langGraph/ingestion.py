import os
from typing import TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
PDF_PATH = "../Gate Data Science And AI.pdf"
INDEX_PATH = "faiss_index"

# --- STATE DEFINITION ---
class IngestionState(TypedDict):
    """The state of our ingestion graph."""
    pdf_path: str
    documents: List
    chunks: List
    is_indexed: bool

# --- NODE 1: LOAD PDF ---
def load_pdf_node(state: IngestionState):
    print(f"📄 [Node: Load] Loading PDF from {state['pdf_path']}...")
    if not os.path.exists(state['pdf_path']):
        raise FileNotFoundError(f"PDF not found at {state['pdf_path']}")
    
    loader = PyPDFLoader(state['pdf_path'])
    documents = loader.load()
    return {"documents": documents}

# --- NODE 2: SPLIT TEXT ---
def split_text_node(state: IngestionState):
    print("✂️ [Node: Split] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(state["documents"])
    return {"chunks": chunks}

# --- NODE 3: BUILD INDEX ---
def build_index_node(state: IngestionState):
    print("🧠 [Node: Index] Generating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store from chunks
    vector_store = FAISS.from_documents(state["chunks"], embeddings)
    
    # Save locally
    print(f"💾 [Node: Index] Saving vector store to {INDEX_PATH}...")
    # Create directory if it doesn't exist (actually FAISS handles this, but good to be safe)
    vector_store.save_local(INDEX_PATH)
    
    return {"is_indexed": True}

# --- GRAPH CONSTRUCTION ---
def build_ingestion_graph():
    workflow = StateGraph(IngestionState)

    # 1. Add Nodes
    workflow.add_node("load_pdf", load_pdf_node)
    workflow.add_node("split_text", split_text_node)
    workflow.add_node("build_index", build_index_node)

    # 2. Add Edges
    workflow.set_entry_point("load_pdf")
    workflow.add_edge("load_pdf", "split_text")
    workflow.add_edge("split_text", "build_index")
    workflow.add_edge("build_index", END)

    # 3. Compile
    return workflow.compile()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    app = build_ingestion_graph()
    
    initial_state = {
        "pdf_path": PDF_PATH,
        "documents": [],
        "chunks": [],
        "is_indexed": False
    }

    print("🚀 Starting LangGraph Ingestion Pipeline...")
    try:
        final_state = app.invoke(initial_state)
        if final_state["is_indexed"]:
            print("✅ Ingestion successfully completed via LangGraph!")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
