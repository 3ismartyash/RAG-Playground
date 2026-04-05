# Learn RAG: Master PDF Question Answering

This project is a multi-stage journey to mastering **RAG (Retrieval-Augmented Generation)**. We will build the same PDF-based AI application in three different ways, moving from raw implementation to high-level framework orchestration.

---

## 🗺️ Learning Path

### Stage 1: [No-Framework Approach](./no_framework/)
Understand the low-level mechanics of PDF chunking, embedding generation using local models, and similarity search logic with **FAISS**. This is built using raw Python logic and calls only the final model from Google's Gemini SDK.

### Stage 2: [LangChain Implementation](./langchain/) (Coming Soon)
A more abstracted approach using **LangChain's** modular chains and loaders. We'll simplify the code while still maintaining full control over the retrieval process.

### Stage 3: [LangGraph Workflow](./langGraph/) (Coming Soon)
Advanced implementation using **LangGraph** to model the RAG pipeline as a stateful, cyclic graph. This will introduce more complex reasoning, error-correction, and agents.

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
├── Readme.Md               # Project Overview (This file)
├── Gate Data Science And AI.pdf  # Source data
├── no_framework/           # Stage 1: Low-level implementation
├── langchain/              # Stage 2: Abstracted chain implementation
└── langGraph/              # Stage 3: Stateful graph/agent implementation
```
