import os
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
INDEX_PATH = "faiss_index"
GEMINI_MODEL = "gemini-1.5-flash"

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    """The state of the graph."""
    input: str
    documents: List[str]
    answer: str

# --- NODE 1: RETRIEVE ---
def retrieve_node(state: AgentState):
    print("🔍 [Node: Retrieve] Fetching documents from FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top 3 documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["input"])
    
    # Extract text from documents
    doc_texts = [d.page_content for d in docs]
    
    return {"documents": doc_texts}

# --- NODE 2: GENERATE ---
def generate_node(state: AgentState):
    print("✨ [Node: Generate] Creating answer using Gemini Pro...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    
    context = "\n---\n".join(state["documents"])
    prompt = f"""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    CONTEXT:
    {context}

    QUESTION:
    {state["input"]}

    ANSWER:
    """
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- GRAPH CONSTRUCTION ---
def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 2. Add Edges (the sequence)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # 3. Compile
    return workflow.compile()

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print("❌ Error: Vector store index not found. Run ingestion.py first.")
    else:
        app = build_graph()
        
        while True:
            user_input = input("\n💬 Ask anything about the PDF (or 'exit'): ")
            if user_input.lower() == 'exit':
                break

            # Execute the Graph
            inputs = {"input": user_input}
            print("🚀 Starting graph execution...")
            result = app.invoke(inputs)
            
            print(f"\n💡 Answer:\n{result['answer']}")
