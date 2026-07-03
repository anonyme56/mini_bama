import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Assistant documentaire Lite RAG", page_icon="📄")
st.title("Assistant documentaire Lite RAG")
st.caption("Corpus : discours de Barack Obama (2013) — TP Module Cloud, École Hexagone")

# on garde en memoire le nom du document courant entre les interactions
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None


@st.cache_data(ttl=10)
def backend_up():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


if not backend_up():
    st.error(
        f"Le backend FastAPI n'est pas joignable sur {BACKEND_URL}. "
        f"Lancez-le avec : uvicorn backend.main:app --host 0.0.0.0 --port 8000"
    )
    st.stop()

st.sidebar.header("1. Document")

# choix entre le fichier deja present dans data/ ou un upload manuel
default_path = "data/discours_obama_2013_fr.txt"
use_default = st.sidebar.checkbox("Utiliser le fichier fourni (discours_obama_2013_fr.txt)", value=True)

uploaded_file = None
if not use_default:
    uploaded_file = st.sidebar.file_uploader("Charger un document (.txt ou .pdf)", type=["txt", "pdf"])

if use_default:
    if st.sidebar.button("Charger le document fourni"):
        if os.path.exists(default_path):
            with open(default_path, "rb") as f:
                files = {"file": (os.path.basename(default_path), f, "text/plain")}
                r = requests.post(f"{BACKEND_URL}/upload", files=files)
            if r.status_code == 200:
                st.session_state.current_filename = r.json()["filename"]
                st.sidebar.success(f"Document charge : {st.session_state.current_filename}")
            else:
                st.sidebar.error(r.json().get("detail", "erreur upload"))
        else:
            st.sidebar.error(f"Fichier introuvable : {default_path}")
elif uploaded_file is not None:
    if st.sidebar.button("Envoyer ce document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        r = requests.post(f"{BACKEND_URL}/upload", files=files)
        if r.status_code == 200:
            st.session_state.current_filename = r.json()["filename"]
            st.sidebar.success(f"Document charge : {st.session_state.current_filename}")
        else:
            st.sidebar.error(r.json().get("detail", "erreur upload"))

st.sidebar.header("2. Indexation")
chunk_size = st.sidebar.slider("chunk_size", 500, 1200, 800, step=50)
chunk_overlap = st.sidebar.slider("chunk_overlap", 50, 200, 120, step=10)

if st.sidebar.button("Indexer le document"):
    if not st.session_state.current_filename:
        st.sidebar.error("Chargez d'abord un document.")
    else:
        with st.spinner("Indexation en cours..."):
            r = requests.post(
                f"{BACKEND_URL}/index",
                json={
                    "filename": st.session_state.current_filename,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            )
        if r.status_code == 200:
            data = r.json()
            st.sidebar.success(f"{data['num_chunks']} passages indexes")
        else:
            st.sidebar.error(r.json().get("detail", "erreur indexation"))

st.sidebar.header("3. Reglages recherche")
top_k = st.sidebar.slider("nombre de passages (top_k)", 1, 8, 4)

if st.sidebar.button("Reinitialiser l'index"):
    r = requests.post(f"{BACKEND_URL}/reset")
    if r.status_code == 200:
        st.sidebar.info("Index reinitialise.")

st.divider()
st.subheader("Poser une question")

question = st.text_input(
    "Votre question",
    placeholder="Ex : Que dit le discours sur le changement climatique ?",
)

if st.button("Envoyer la question") and question.strip():
    with st.spinner("Recherche et generation de la reponse..."):
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": question, "top_k": top_k})

    if r.status_code != 200:
        st.error(r.json().get("detail", "erreur lors de la question"))
    else:
        data = r.json()
        st.markdown("### Reponse")
        st.write(data["answer"])

        st.markdown("### Sources utilisees")
        if not data["sources"]:
            st.info("Aucune source (document non indexe ou information absente).")
        for src in data["sources"]:
            with st.expander(f"{src['filename']} — passage {src['chunk_index']} (score {src['similarity_score']})"):
                st.write(src["text"])
