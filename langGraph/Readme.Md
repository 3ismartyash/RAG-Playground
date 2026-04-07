# Stage 3: LangGraph Implementation

Welcome to the final stage! We are now using **LangGraph** to model our RAG pipeline as a stateful, modular graph.

---

## 📈 What Changed from Stage 2?

In Stage 2 (LangChain), we used a "Chain" which is a linear, pre-defined sequence of steps:
`User -> Retriever -> Prompt -> LLM -> Answer`.

In **Stage 3 (LangGraph)**, we treat each Step as a **Node** in a graph:
- **`retrieve` Node**: Responsible solely for fetching data.
- **`generate` Node**: Responsible solely for reasoning over the data.

### Why use a Graph?
- **Modularity**: Nodes can be tested independently.
- **Cyclic Logic**: In more advanced versions, if the `generate` node decides it doesn't have enough information, it can send an edge **back** to the `retrieve` node to try again.
- **State Management**: The graph maintains a shared `AgentState` object that flows between nodes, collecting data at each step.

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

## 🏗️ Graph Structure in this Project
```mermaid
graph LR
    Start --> Retrieve_Node
    Retrieve_Node --> Generate_Node
    Generate_Node --> End
```
In this simple implementation, the path is linear, but you're now set up with the foundation to add loops and conditional logic!
