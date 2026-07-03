import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from backend.rag_core import chunk_text, build_prompt
from backend.main import app

client = TestClient(app)


def test_chunk_text_basic():
    # decoupe un texte simple, verifie qu'on obtient bien des morceaux
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_overlap_invalid():
    try:
        chunk_text("bonjour", chunk_size=100, chunk_overlap=100)
        assert False, "aurait du lever une ValueError"
    except ValueError:
        pass


def test_build_prompt_contains_question_and_context():
    passages = [{"chunk_index": 0, "text": "le climat est important"}]
    prompt = build_prompt("Que dit le texte sur le climat ?", passages)
    assert "climat" in prompt
    assert "Que dit le texte sur le climat ?" in prompt


def test_build_prompt_no_passages():
    prompt = build_prompt("question sans reponse", [])
    assert "aucun passage trouve" in prompt


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
