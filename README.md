# RAG Dictionary — Local, Open-Source, End-to-End

This project implements a two-way dictionary using local, free tools:
- Forward lookup: `word -> meaning`
- Reverse lookup: `meaning -> word` (semantic search via embeddings + FAISS)

Stack:
- Orchestration: Apache Airflow
- Vector store: FAISS (local file)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Serving API: FastAPI
- Caching: Redis
- Monitoring: Prometheus + Grafana
All components run locally via Docker Compose.

---

## Setup (Linux / macOS / WSL / Windows with Docker Desktop)

1. Clone the repo and change into directory:
```bash
git clone <this-repo> rag_dictionary
cd rag_dictionary
