import os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import rag_core

# stockage local du document brut, remplace MinIO pour rester simple en Codespaces
STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="Assistant documentaire Lite RAG", version="1.0.0")

# CORS ouvert car Streamlit tourne sur un port different dans Codespaces
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IndexRequest(BaseModel):
    filename: str
    chunk_size: int = rag_core.DEFAULT_CHUNK_SIZE
    chunk_overlap: int = rag_core.DEFAULT_CHUNK_OVERLAP


class AskRequest(BaseModel):
    question: str
    top_k: int = rag_core.DEFAULT_TOP_K


class Source(BaseModel):
    filename: Optional[str]
    chunk_index: Optional[int]
    similarity_score: Optional[float]
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    return {"documents": sorted(os.listdir(STORAGE_DIR))}


def _extract_text_from_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="nom de fichier manquant")
    if not (filename.lower().endswith(".txt") or filename.lower().endswith(".pdf")):
        raise HTTPException(status_code=400, detail="seuls les fichiers .txt et .pdf sont acceptes")

    destination = os.path.join(STORAGE_DIR, filename)
    content = await file.read()
    with open(destination, "wb") as f:
        f.write(content)

    return {"filename": filename, "size_bytes": len(content), "stored_at": destination}


@app.post("/index")
def index_document(payload: IndexRequest):
    path = os.path.join(STORAGE_DIR, payload.filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"document '{payload.filename}' introuvable")

    if path.lower().endswith(".pdf"):
        text = _extract_text_from_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if not text.strip():
        raise HTTPException(status_code=400, detail="document vide ou illisible")

    try:
        result = rag_core.index_document(
            filename=payload.filename,
            text=text,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return result


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question vide")

    try:
        result = rag_core.answer_question(payload.question, top_k=payload.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return result


@app.post("/reset")
def reset():
    # reinitialise l'index vectoriel (bonus)
    rag_core.reset_collection()
    return {"status": "index reinitialise"}
