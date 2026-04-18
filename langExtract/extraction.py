import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# --- CONFIG ---
load_dotenv(find_dotenv())
# Resolve index path relative to this script file, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index")
GEMINI_MODEL = "gemini-flash-latest"

# --- STRATEGY: Manual JSON parsing ---
# langchain-google-genai v2.0.0 uses Gemini's function-calling mode for
# with_structured_output(), which CANNOT serialize nested List[Model] schemas
# (Gemini requires an explicit 'items' descriptor that the old SDK omits).
#
# Solution: ask Gemini to return a plain JSON string, then parse it ourselves
# with json.loads(). This is fully compatible with v2.0.0 and gives us
# complete control over the schema description.

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string", "description": "Main subject / topic name"},
                    "sub_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific concepts or sub-topics covered"
                    },
                    "importance_summary": {"type": "string", "description": "Why this topic matters for the exam"}
                },
                "required": ["topic_name", "sub_topics"]
            }
        }
    },
    "required": ["topics"]
}


def run_extraction():
    print("Loading vector store for extraction...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(INDEX_PATH):
        print("Error: Vector store index not found. Run ingestion.py first.")
        return

    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # 1. Fetch relevant context for "Syllabus"
    # We retrieve more chunks than usual to get a broader view for extraction
    print("Searching for syllabus content...")
    docs = vector_store.similarity_search("What is the syllabus for Data Science and AI?", k=5)
    context = "\n---\n".join([d.page_content for d in docs])

    # 2. Setup LLM (plain, no structured output wrapper)
    print("Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    # 3. Build prompt with inline schema description
    schema_hint = json.dumps(EXTRACTION_SCHEMA, indent=2)
    prompt = f"""
You are an expert data extractor. Extract the main topics and sub-topics from the GATE Data Science and AI syllabus in the context below.

CONTEXT:
{context}

Return ONLY a single JSON object that exactly matches this schema (no markdown, no extra text):
{schema_hint}

Extract as many topics as the context allows.
"""

    # 4. Invoke and parse
    print("Extracting structured data...")
    raw_response = llm.invoke(prompt)

    # Strip possible markdown code fences Gemini sometimes adds
    raw_text = raw_response.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    result = json.loads(raw_text)

    # 5. Display Results
    print("\nEXTRACTION COMPLETE!")
    print("-" * 50)
    for topic in result.get("topics", []):
        print(f"\n  TOPIC: {topic['topic_name']}")
        sub = ", ".join(topic.get("sub_topics", []))
        print(f"  Concepts: {sub}")
        note = topic.get("importance_summary")
        if note:
            print(f"  Note: {note}")
    print("-" * 50)


if __name__ == "__main__":
    run_extraction()
