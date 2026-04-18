# Learn RAG: Master PDF Question Answering

This project is a multi-stage journey to mastering **RAG (Retrieval-Augmented Generation)**. We will build the same PDF-based AI application in three different ways, moving from raw implementation to high-level framework orchestration.

---

## 🗺️ Learning Path

### Stage 1: [No-Framework Approach](./no_framework/)
Understand the low-level mechanics of PDF chunking, embedding generation using local models, and similarity search logic with **FAISS**. This is built using raw Python logic and calls only the final model from Google's Gemini SDK.

### Stage 2: [LangChain Implementation](./langchain/)
A more abstracted approach using **LangChain's** modular chains and loaders. We simplify the code while still maintaining full control over the retrieval process.

### Stage 3: [LangGraph Workflow](./langGraph/)
Advanced implementation using **LangGraph** to model the RAG pipeline as a stateful, cyclic graph. This introduces more complex reasoning, error-correction, and agents using a node-based architecture.

### Stage 4: [LangChain Extraction](./langExtract/)
Introduction to **Information Extraction**. Instead of chatting with the PDF, we use LangChain's structured output features to transform unstructured text into clean, machine-readable JSON data.

---

## 🔍 Database Inspection & Raw Vector Analysis

To help you understand exactly how your data is stored "under the hood," we've added several diagnostic tools in the root directory:

*   **[`inspect_pickle.py`](./inspect_pickle.py)**: Loads the serialized text chunks from the `no_framework` directory so you can read the raw text after splitting.
*   **[`inspect_faiss.py`](./inspect_faiss.py)**: Automatically detects and loads any of your three FAISS databases (No-Framework, LangChain, or LangGraph) to show you the number of vectors and their corresponding text/metadata.
*   **[`raw_vectors.py`](./raw_vectors.py)**: The "deep dive" tool. It extracts the actual **reconstructed vectors** (NumPy arrays of size 384) from the FAISS binary so you can see the mathematical "coordinates" of your data.

---

## 🛠️ Project Setup

All stages share the same core dependencies. It is recommended to use a single virtual environment for the entire project.

### 1. Create a Virtual Environment
Run the following commands in your terminal from this root folder:

```bash
# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
You only need to set your Google Gemini API Key once. Create a `.env` file in the root directory (or use the one in each folder):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Requirements Overview
The project uses the following key libraries:
- `pypdf`: To read and extract text from the PDF.
- `faiss-cpu`: Facebook's high-performance similarity search library.
- `sentence-transformers`: Local text embedding models from Hugging Face.
- `google-generativeai`: Access to Google's Gemini Pro LLMs.
- `langchain` & `langgraph`: Orchestration frameworks for advanced stages.

---

## 📂 Repository Structure
```text
Sample_prj/
├── venv/                   # Shared virtual environment
├── requirements.txt        # Shared dependencies
├── README.md               # Project Overview (This file)
├── Gate Data Science And AI.pdf  # Source data
│
├── inspect_faiss.py        # Diagnostic: Inspect FAISS text/metadata
├── inspect_pickle.py       # Diagnostic: Inspect raw text chunks
├── raw_vectors.py          # Diagnostic: View actual mathematical vectors
│
├── no_framework/           # Stage 1: Low-level implementation
├── langchain/              # Stage 2: Abstracted chain implementation
├── langGraph/              # Stage 3: Stateful graph/agent implementation
└── langExtract/            # Stage 4: Structured Information Extraction
```
