# Assistant documentaire Lite RAG

TP final module Cloud — École Hexagone. Petite application qui répond à des questions
en français sur le discours d'investiture de Barack Obama (2013), en s'appuyant
uniquement sur le contenu du document (RAG = Retrieval-Augmented Generation).

## Objectif du projet

Construire une chaîne RAG légère de bout en bout : chargement d'un document,
découpage en passages, embeddings, stockage vectoriel, recherche, génération de
réponse par un petit LLM local, et affichage des sources utilisées. Ce n'est pas une
application de production, mais une démonstration fonctionnelle du principe.

## Fonctionnement du RAG (résumé)

```
Document -> Découpage -> Embeddings -> Base vectorielle -> Recherche
-> Prompt enrichi -> Réponse + sources
```

Le document est découpé en passages courts. Chaque passage est transformé en vecteur
(embedding) et stocké dans ChromaDB. Quand l'utilisateur pose une question, elle est
elle aussi transformée en vecteur, ce qui permet de retrouver les passages les plus
proches. Ces passages sont injectés dans un prompt envoyé au LLM, qui doit répondre
uniquement à partir de ce contexte. S'il ne trouve rien de pertinent, l'application le
signale clairement plutôt que d'inventer une réponse.

## Outils utilisés

| Besoin                  | Outil                                                            |
|--------------------------|-------------------------------------------------------------------|
| Environnement de dev     | GitHub Codespaces                                                  |
| Interface utilisateur    | Streamlit                                                          |
| Backend API              | FastAPI                                                            |
| Stockage du document brut| Stockage local (dossier `storage/`) — voir justification ci-dessous|
| Base vectorielle         | ChromaDB (persistant, dossier `chroma_db/`)                        |
| Modèle d'embeddings      | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2        |
| LLM local léger          | Ollama, modèle qwen2.5:0.5b                                        |
| CI                       | GitHub Actions                                                     |

### Pourquoi pas MinIO ?

Le sujet recommande MinIO pour le stockage du document brut. Pour rester simple à
lancer en Codespaces sans service supplémentaire à configurer (credentials, bucket,
port), le document uploadé est simplement écrit dans le dossier `storage/` du
backend. Le rôle est le même que MinIO (stocker le fichier brut et le retrouver par
nom de fichier) : c'est un choix d'implémentation, pas un changement d'architecture.
Passer à MinIO reviendrait à remplacer les fonctions `open()`/`read()` du fichier
`backend/main.py` par des appels au SDK MinIO (boto3), sans toucher au reste de la
chaîne RAG.

## Structure du projet

```
rag-lite-obama/
├── app.py                     # interface Streamlit
├── backend/
│   ├── main.py                 # API FastAPI (/health /upload /index /ask /reset)
│   └── rag_core.py             # chunking, embeddings, ChromaDB, prompt, appel Ollama
├── data/
│   └── discours_obama_2013_fr.txt   # document fourni pour le TP
├── storage/                    # documents uploadés (stockage local, remplace MinIO)
├── chroma_db/                  # base vectorielle persistante (généré automatiquement)
├── tests/
│   └── test_rag.py             # tests unitaires (chunking, prompt, endpoint /health)
├── .github/workflows/ci.yml    # pipeline CI (installe + lance les tests)
├── requirements.txt
└── README.md
```

## Lancer avec Docker (alternative recommandée pour un déploiement ailleurs que Codespaces)

Le projet peut être lancé entièrement avec Docker, sans rien installer en local
(ni Python, ni Ollama) :

```bash
docker compose up --build
```

Ça démarre 3 services : `ollama` (le serveur LLM), `backend` (FastAPI) et
`streamlit` (l'interface). Le modèle `qwen2.5:0.5b` est téléchargé automatiquement
au premier lancement par le service `ollama-pull`, puis conservé dans un volume
Docker (pas besoin de le retélécharger aux lancements suivants).

Une fois les services démarrés, ouvre http://localhost:8501.

Fichiers concernés : `Dockerfile.backend`, `Dockerfile.streamlit`, `docker-compose.yml`.

C'est l'option la plus simple si tu veux déployer le projet ailleurs que dans
Codespaces (VPS, autre machine...) : Docker embarque tout, il n'y a pas besoin de
réinstaller Ollama ou les dépendances Python à la main sur la machine cible.

## Installation (sans Docker)

```bash
pip install -r requirements.txt
```

## Installer et lancer Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:0.5b
ollama list
```

Vous pouvez tester le modèle avant de lancer l'application :

```bash
ollama run qwen2.5:0.5b
```

## Lancer l'application

Deux processus séparés sont nécessaires : le backend FastAPI et l'interface
Streamlit.

Terminal 1 — backend :

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — interface :

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Dans Codespaces, rendez le port 8501 public (et le port 8000 si vous voulez tester
l'API directement), puis ouvrez le lien fourni par Codespaces.

## Indexer le document

Depuis l'interface Streamlit :
1. Cochez "Utiliser le fichier fourni" (ou uploadez votre propre `.txt`/`.pdf`).
2. Cliquez sur "Charger le document fourni".
3. Ajustez éventuellement `chunk_size` / `chunk_overlap` dans la barre latérale.
4. Cliquez sur "Indexer le document".

Paramètres par défaut : `chunk_size=800`, `chunk_overlap=120`, `top_k=4`, conformes
aux valeurs conseillées dans le sujet.

## Exemple de question

> Que dit le discours sur le changement climatique ?

L'application retourne une réponse synthétique, suivie des passages source utilisés
(nom du fichier, numéro du passage, score de similarité), consultables dans des
encarts dépliables sous la réponse.

Si vous posez une question sans rapport avec le document (ex : "Quelle est la
capitale du Japon ?"), l'application doit répondre qu'elle ne trouve pas
l'information dans le document fourni.

## API (si utilisée directement, sans Streamlit)

```
GET  /health              -> {"status": "ok"}
GET  /documents            -> liste des documents stockés
POST /upload (multipart)    -> uploade un fichier .txt ou .pdf
POST /index                  -> {"filename": "...", "chunk_size": 800, "chunk_overlap": 120}
POST /ask                     -> {"question": "...", "top_k": 4}
POST /reset                    -> réinitialise l'index vectoriel
```

## Tests

```bash
pytest tests/ -v
```

Les tests couvrent le découpage en chunks (cas normal, texte vide, paramètres
invalides), la construction du prompt, et la disponibilité de l'endpoint `/health`.
Ils ne dépendent pas d'Ollama (les tests ne font pas d'appel au LLM).

## Limites connues

- Temps de réponse parfois long avec le modèle léger `qwen2.5:0.5b`.
- Qualité de réponse imparfaite : le modèle est volontairement petit.
- Un seul document à la fois par défaut (le fichier fourni pour le TP).
- Pas d'authentification, pas de monitoring, pas de déploiement production.
- Interface volontairement simple, sans mise en forme avancée.
- L'extraction PDF (bonus) fonctionne uniquement sur des PDF texte, pas sur des
  documents scannés (pas d'OCR).

## Bonus implémentés

- Support PDF (extraction via `pypdf`, si le texte est extractible).
- Réglage du `top_k` (nombre de passages récupérés) depuis l'interface.
- Affichage du score de similarité pour chaque source.
- Bouton de réinitialisation de l'index vectoriel.
- Réglage de `chunk_size` / `chunk_overlap` depuis l'interface.
- Dockerisation complète (`Dockerfile.backend`, `Dockerfile.streamlit`, `docker-compose.yml`) pour déployer le projet en dehors de Codespaces sans rien installer en local.
