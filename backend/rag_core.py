import os
import uuid
from typing import List, Dict, Any

import requests
import chromadb
from sentence_transformers import SentenceTransformer

# chemins et config, modifiables via variables d'env
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4

PROMPT_TEMPLATE = """Tu es un assistant documentaire.
Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas présente dans le contexte, réponds :
"Je ne trouve pas cette information dans le document fourni."

Contexte :
{context}

Question :
{question}

Réponse :"""


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    # decoupe le texte en passages avec chevauchement
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit etre inferieur a chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # on essaie de couper sur un espace pour ne pas couper un mot en deux
        if end < n:
            space = text.rfind(" ", start, end)
            if space != -1 and space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0

    return chunks


_embedder = None


def get_embedder():
    # charge le modele d'embeddings une seule fois (cache global)
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts):
    model = get_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


_chroma_client = None


def get_chroma_collection():
    # client persistant chroma, cree une seule fois
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection():
    # supprime puis recree la collection (bouton reset, bonus)
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        _chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def index_document(filename, text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("document vide, rien a indexer")

    embeddings = embed_texts(chunks)
    collection = get_chroma_collection()

    ids = [f"{filename}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i, "excerpt": c[:200]} for i, c in enumerate(chunks)]

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {"filename": filename, "num_chunks": len(chunks)}


def search(question, top_k=DEFAULT_TOP_K):
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))

    passages = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # distance chroma -> score de similarite approximatif entre 0 et 1
        score = round(1 / (1 + dist), 4)
        passages.append({
            "text": doc,
            "filename": meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "similarity_score": score,
        })

    return passages


def build_prompt(question, passages):
    context = "\n\n".join(f"[Passage {p['chunk_index']}] {p['text']}" for p in passages)
    if not context:
        context = "(aucun passage trouve)"
    return PROMPT_TEMPLATE.format(context=context, question=question)


def call_ollama(prompt, model=OLLAMA_MODEL, timeout=120):
    # appel simple a l'API locale d'ollama (pas de streaming)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Ollama injoignable, verifier 'ollama serve' et 'ollama pull {model}'"
        ) from e


def answer_question(question, top_k=DEFAULT_TOP_K):
    passages = search(question, top_k=top_k)

    if not passages:
        return {"answer": "Je ne trouve pas cette information dans le document fourni.", "sources": []}

    prompt = build_prompt(question, passages)
    answer = call_ollama(prompt)

    return {"answer": answer, "sources": passages}
