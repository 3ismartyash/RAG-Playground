# Stage 2: LangChain Implementation

Welcome to Stage 2 of the RAG journey! In this folder, we move away from manual text splitting and index management to using **LangChain**, the industry-standard orchestration framework.

---

## 🔄 What Changed from Stage 1?

In Stage 1 (No-Framework), we had to manually:
- Parse PDF pages to strings.
- Write math for chunking logic.
- Handle metadata mapping in separate pickle files for FAISS.
- Format strings the the LLM prompt manually.

In **Stage 2 (LangChain)**, we use high-level abstractions:
- **`PyPDFLoader`**: Automatically handles page splitting and basic metadata.
- **`RecursiveCharacterTextSplitter`**: Smarter splitting that tries to keep sentences and paragraphs together.
- **`FAISS` (Wrapper)**: LangChain’s FAISS integration stores the text *inside* the vector store object, so we don't need `chunks.pkl`.
- **`create_retrieval_chain`**: Connects our vector store to our LLM with a single, clean workflow.

---

## 🛠️ How to Run

1.  **Activate Environment**:
    ```powershell
    ..\venv\Scripts\activate
    ```
2.  **Add API Key**: Ensure your `.env` is set up with `GOOGLE_API_KEY`. (You can copy the one from `no_framework`)
3.  **Run Ingestion**:
    ```powershell
    python ingestion.py
    ```
4.  **Ask Questions**:
    ```powershell
    python query.py
    ```

---

## 📦 Key Concepts Introduced
- **Retriever**: An interface that takes a string query and returns the most relevant documents.
- **Tools/Loaders**: specialized classes to handle different file formats easily.
- **Chains**: The "glue" logic that connects retrievers, prompts, and LLMs.
