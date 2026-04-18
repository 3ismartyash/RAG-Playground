# Stage 4: LangChain Extraction

Welcome to Stage 4! In the previous stages, we focused on **Question Answering** (Chatting with your PDF). In this stage, we explore another powerful use case for LLMs: **Information Extraction**.

---

## 📊 What is Information Extraction?

Unlike standard RAG which returns a conversational answer, **Extraction** transforms unstructured text (sentences, paragraphs) into **Structured Data** (JSON, Pydantic objects).

### Why use Extraction?
- **Data Integration**: Turn a PDF syllabus into a database-ready list of topics.
- **Consistency**: Ensure the LLM always returns data in the exact same format.
- **Automation**: Automatically pull dates, names, or values from thousands of documents without manual searching.

---

## 🛠️ How to Run

1.  **Activate Environment**:
    ```powershell
    ..\venv\Scripts\activate
    ```
2.  **Run Ingestion**:
    ```powershell
    python ingestion.py
    ```
3.  **Run Extraction**:
    ```powershell
    python extraction.py
    ```

---

## 🏗️ Technical Highlights

- **`Pydantic`**: We define a class `ExamTopic` to act as a "blueprint" for the data we want.
- **`.with_structured_output()`**: A specialized LangChain method that forces the LLM (Gemini) to follow our Pydantic schema exactly.
- **Focused Retrieval**: Instead of answering a specific user question, we retrieve context related to the broad concept of "Syllabus" to provide enough data for the extraction logic.

---

## 💡 Pro Tip
Try changing the schema in `extraction.py` to extract something else (like "Suggested Books" or "Weightage") to see how flexible this system is!
