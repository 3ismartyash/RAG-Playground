# No-Framework RAG Pipeline: A Learning Journey

This project aims to build a complete **Retrieval-Augmented Generation (RAG)** pipeline from scratch using only core Python libraries and standalone packages. No high-level orchestration frameworks like **LangChain** or **LangGraph** are used. This approach is designed for deep learning and understanding the internal mechanics of AI applications.

---

## 🎯 Project Overview
We are building a pipeline that allows users to query a specific document (in this case, the `Gate Data Science And AI.pdf`) and get accurate, context-aware answers using Google's Gemini API.

---

## 🏗️ Architecture & Pipeline Steps

### 1. Document Ingestion (PDF Parsing)
*   **Purpose**: Convert the raw binary PDF file into a processable text format.
*   **Package**: `pypdf`
*   **Logic**: Iterate through pages and extract text strings.

### 2. Text Chunking (Manual Logic)
*   **Purpose**: Large documents exceed LLM context windows. We must split the text into manageable "chunks."
*   **Logic**: Use a sliding window approach (e.g., 500–1000 characters per chunk with a 10% overlap to preserve context).

### 3. Embedding Generation
*   **Purpose**: Convert text chunks into numerical vectors (embeddings) that represent semantic meaning.
*   **Model**: `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
*   **Logic**: Each text chunk is passed through the model to output a high-dimensional vector.

### 4. Vector Storage (FAISS)
*   **Purpose**: Efficiently store and search through thousands of text vectors.
*   **Package**: `faiss-cpu` (Facebook AI Similarity Search).
*   **Logic**: Create an index (L2 distance or Inner Product) and add the embedding vectors to it.

### 5. User Query Processing
*   **Purpose**: Transform the user's natural language question into the same vector space as the document.
*   **Logic**: The user question is embedded using the same `sentence-transformers` model.

### 6. Semantic Search (Retrieval)
*   **Purpose**: Find the most relevant parts of the document for the given question.
*   **Logic**: Use FAISS to find the "Top K" (e.g., Top 3) closest vectors to the query vector.

### 7. Generative AI (LLM Response)
*   **Purpose**: Synthesize the retrieved facts into a human-readable answer.
*   **API**: `google-generativeai` (Google Gemini Pro).
*   **Logic**: Construct a prompt: `Context: [Retrieved Chunks] \n Question: [User Query] \n Answer:`

---

## 📦 Requirements & Packages

Install the following core packages:

```bash
pip install pypdf faiss-cpu sentence-transformers google-generativeai python-dotenv
```

| Package | Role |
| :--- | :--- |
| **pypdf** | PDF text extraction |
| **faiss-cpu** | Vector database and similarity search |
| **sentence-transformers** | Local embedding generation (no API needed for vectors) |
| **google-generativeai** | LLM for the final answer synthesis |
| **python-dotenv** | Managing your Google API Key securely |

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## 📂 Project Structure

```text
no_framework/
├── .env                # API Keys
├── Readme.Md           # Project Roadmap
├── ingest.py           # Logic for PDF -> Vectors (Index creation)
├── query.py            # Logic for Query -> Search -> LLM
└── pipeline.py         # Main entry point combining ingestion and querying
```

---

## 🚀 How to Run (Plan)

1.  **Extract & Embed**: Run the ingestion script to process `../Gate Data Science And AI.pdf`.
2.  **Save Index**: Store the FAISS index locally so you don't have to re-process the PDF every time.
3.  **Query**: Run the query script to ask questions about Data Science & AI.