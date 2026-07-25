"""
FastAPI backend for the document-based chatbot.

Endpoints:
- POST /admin/upload   -> admin uploads a document (pdf/docx/txt), it gets
                           chunked, embedded, and stored in the vector DB
- GET  /admin/documents -> list indexed documents (admin visibility)
- DELETE /admin/documents/{doc_name} -> remove a document from the index
- POST /chat            -> user asks a question, gets a grounded answer

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
import glob

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Document Chatbot API")

# Allow the simple frontend (served separately or via file://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    doc_name: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    """Admin endpoint: upload a document to be chunked, embedded, and indexed."""
    allowed_ext = {"pdf", "docx", "txt"}
    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = rag.ingest_document(save_path, doc_name=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return result


@app.get("/admin/documents")
async def get_documents():
    """List all documents currently indexed in the vector store."""
    return rag.list_documents()


@app.delete("/admin/documents/{doc_name}")
async def remove_document(doc_name: str):
    """Delete a document (and its chunks) from the vector store."""
    deleted = rag.delete_document(doc_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_name": doc_name, "chunks_deleted": deleted}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """User endpoint: ask a natural-language question, get a grounded answer."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = rag.answer_query(request.query, doc_name=request.doc_name)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

@app.on_event("startup")
async def auto_ingest_permanent_documents():
    """On server startup, index any documents in /documents that aren't already indexed."""
    if not os.path.isdir(DOCUMENTS_DIR):
        return

    already_indexed = {doc["doc_name"] for doc in rag.list_documents()}

    for file_path in glob.glob(os.path.join(DOCUMENTS_DIR, "*")):
        filename = os.path.basename(file_path)
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext not in {"pdf", "docx", "txt"}:
            continue
        if filename in already_indexed:
            print(f"Already indexed: {filename}")
            continue
        print(f"Auto-indexing permanent document: {filename}")
        try:
            result = rag.ingest_document(file_path, doc_name=filename)
            print(f"Indexed {filename}: {result}")
        except Exception as e:
            print(f"Failed to auto-index {filename}: {e}")


if os.path.isdir(DOCUMENTS_DIR):
    app.mount("/documents", StaticFiles(directory=DOCUMENTS_DIR), name="documents")

if os.path.isdir(UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# Serve the simple frontend directly from FastAPI for convenience
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
