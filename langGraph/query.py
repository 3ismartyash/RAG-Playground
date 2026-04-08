import os
from typing import List, TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
INDEX_PATH = "faiss_index"
GEMINI_MODEL = "gemini-flash-latest"

# --- STATE DEFINITION ---gemini-flash-latest
class AgentState(TypedDict):
    """The state of our RAG graph."""
    input: str
    documents: List[str]
    answer: str
    is_relevant: bool

# --- NODE 1: RETRIEVE ---
def retrieve_node(state: AgentState):
    print("🔍 [Node: Retrieve] Fetching documents from FAISS...")
    # Load same embeddings used in ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Vector index not found at {INDEX_PATH}. Run ingestion.py first!")

    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top 3 documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["input"])
    
    # Extract text from documents
    doc_texts = [d.page_content for d in docs]
    
    return {"documents": doc_texts}

# --- NODE 2: GRADE RELEVANCE ---
def grade_documents_node(state: AgentState):
    print("🔬 [Node: Grade] Checking if retrieved info is relevant to the question...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    
    context = "\n---\n".join(state["documents"])
    prompt = f"""
    You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    
    USER QUESTION: {state["input"]}
    RETRIEVED CONTEXT: {context}
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Respond with only 'yes' or 'no'.
    """
    
    response = llm.invoke(prompt)
    is_relevant = response.content.strip().lower() == 'yes'
    
    return {"is_relevant": is_relevant}

# --- NODE 3: GENERATE (Success Path) ---
def generate_node(state: AgentState):
    print("✨ [Node: Generate] Creating answer using Gemini Pro...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    
    context = "\n---\n".join(state["documents"])
    prompt = f"""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    Keep the answer concise (maximum 3 sentences).
    
    CONTEXT:
    {context}
    
    QUESTION:
    {state["input"]}
    
    ANSWER:
    """
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- NODE 4: FALLBACK (No Relevance Path) ---
def fallback_node(state: AgentState):
    print("⚠️ [Node: Fallback] Retrieved context was not relevant.")
    return {"answer": "I'm sorry, but looking through the provided documents, I couldn't find specific information to answer that question accurately."}

# --- CONDITIONAL ROUTER ---
def decide_to_generate(state: AgentState) -> Literal["generate", "fallback"]:
    if state["is_relevant"]:
        return "generate"
    else:
        return "fallback"

# --- GRAPH CONSTRUCTION ---
def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # 2. Add Edges (the logic flow)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # Conditional Edge: If grade is 'yes', go to generate. Else fallback.
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # 3. Compile
    return workflow.compile()

# --- MAIN LOOP ---
if __name__ == "__main__":
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
