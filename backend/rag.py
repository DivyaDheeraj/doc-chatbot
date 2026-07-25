"""
Core RAG (Retrieval-Augmented Generation) logic — powered by Google Gemini.

Handles:
- Extracting text from uploaded documents (PDF, DOCX, TXT)
- Chunking text into overlapping windows
- Generating embeddings via Gemini's embedding model
- Storing/retrieving chunks from a local ChromaDB vector store
- Building a grounded prompt and calling Gemini for the answer
"""

import os
import uuid
from typing import List, Dict

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---- Configuration -------------------------------------------------------

load_dotenv()  # reads GEMINI_API_KEY from .env
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-flash-latest"   # fast + accurate, good for this use case

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 8

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
COLLECTION_NAME = "documents"

# ---- Vector store setup ---------------------------------------------------

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# ---- Text extraction -------------------------------------------------------

def extract_text(file_path: str) -> str:
    ext = file_path.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == "docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---- Chunking ---------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap

    return chunks


# ---- Embedding ----------------------------------------------------------------

import time

def embed_texts(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Generate embeddings using Gemini. Batches + rate-limits to stay within free tier limits."""
    vectors = []
    batch_size = 20  # smaller batches to avoid quota spikes
    delay_between_batches = 2  # seconds

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=EMBED_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                vectors.extend(result["embedding"])
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)  # back off: 15s, 30s, 45s...
                    print(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        time.sleep(delay_between_batches)

    return vectors

def embed_query(text: str) -> List[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


# ---- Ingestion ----------------------------------------------------------------

def ingest_document(file_path: str, doc_name: str) -> Dict:
    text = extract_text(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return {"doc_name": doc_name, "chunks_added": 0, "warning": "No extractable text found."}

    embeddings = embed_texts(chunks)
    ids = [f"{doc_name}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"doc_name": doc_name, "chunk_index": i} for i in range(len(chunks))]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {"doc_name": doc_name, "chunks_added": len(chunks)}


def list_documents() -> List[Dict]:
    data = _collection.get(include=["metadatas"])
    counts: Dict[str, int] = {}
    for meta in data.get("metadatas", []):
        name = meta.get("doc_name", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return [{"doc_name": name, "chunks": count} for name, count in counts.items()]


def delete_document(doc_name: str) -> int:
    existing = _collection.get(where={"doc_name": doc_name}, include=[])
    ids = existing.get("ids", [])
    if ids:
        _collection.delete(ids=ids)
    return len(ids)


# ---- Retrieval + generation -----------------------------------------------------

SYSTEM_PROMPT = """You are a document-based assistant. You must answer ONLY using the \
CONTEXT provided below, which was retrieved from the uploaded document(s).

Rules you must strictly follow:
1. Answer only from the given CONTEXT. Do not use outside knowledge.
2. Do not make assumptions or infer information that is not explicitly present.
3. If the answer is not found in the CONTEXT, respond exactly with: "Information not available."
4. Be clear and concise. Do not pad the answer with speculation.
5. Understand implied or related phrasing in the question (for example, "education details" \
should match sections like degrees, certifications, or study history even if worded differently \
in the document) — but only if that information is genuinely present in the CONTEXT.
"""


def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Dict]:
    query_embedding = embed_query(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": doc,
            "doc_name": meta.get("doc_name"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
        })
    return chunks


def answer_query(query: str, top_k: int = TOP_K) -> Dict:
    chunks = retrieve_chunks(query, top_k=top_k)

    if not chunks:
        return {"answer": "Information not available.", "sources": []}

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['doc_name']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nANSWER:"

    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 5s, 10s
                print(f"Chat rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

    return {
        "answer": response.text.strip(),
        "sources": [
            {"doc_name": c["doc_name"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    }