import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
INDEX_PATH = "faiss_index"
GEMINI_MODEL = "gemini-flash-latest" # The updated model name

def query_system():
    # 1. Load Embeddings and Vector Store
    print("📦 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # allow_dangerous_deserialization=True is required for loading local pickle-based FAISS
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Setup Retriever
    retriever = vector_store.as_retriever()

    # 3. Setup LLM
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

    # 4. Define Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 5. Create Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 6. Chat Loop
    while True:
        user_input = input("\n💬 Ask anything about the PDF (or 'exit'): ")
        if user_input.lower() == 'exit':
            break

        print("🔍 Searching and thinking...")
        response = rag_chain.invoke({"input": user_input})
        
        print(f"\n💡 Answer:\n{response['answer']}")

if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print("❌ Error: Vector store index not found. Run ingestion.py first.")
    else:
        query_system()
