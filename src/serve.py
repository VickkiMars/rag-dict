# src/serve.py

from fastapi import FastAPI, Query
from src.cache import kv_get, kv_set, json_get, json_set
from src.embeddings import embed_texts, get_model
from src.vector_store import load_faiss, search_index, prepare_data_and_index
from src.reranker import bm25_rerank
from src.config import (
    DICT_JSON, FAISS_PATH, EMBEDDING_MODEL, TOP_K, REDIS_PREFIX
)
from prometheus_fastapi_instrumentator import Instrumentator
import json
from src.dict_trie import load_trie, prefix_search, build_trie_from_json
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path

static_dir = Path(__file__).resolve().parent / "frontend"

# Initialize FastAPI
app = FastAPI(title="RAG Dictionary API", version="1.0")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# --- Load Data and FAISS Index ---
trie = None
with open(DICT_JSON, "r", encoding="utf-8") as f:
    dictionary_data = json.load(f)
faiss_index = None
model = None

# --- Metrics Setup ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.on_event("startup")
def startup_event():
    global trie, faiss_index, model
    model = get_model(EMBEDDING_MODEL)
    if not os.path.exists("data/dictionary.trie"):
        trie = build_trie_from_json(DICT_JSON)
    trie = load_trie("data/dictionary.trie")

    if not os.path.exists(FAISS_PATH):
        faiss_index, _ = prepare_data_and_index(EMBEDDING_MODEL)
    else:
        faiss_index = load_faiss(FAISS_PATH)

# --- Utility ---
def make_cache_key(prefix: str, query: str) -> str:
    return f"{REDIS_PREFIX}{prefix}:{query.strip().lower()}"

# --- API Endpoints ---

@app.get("/define")
def define(word: str = Query(..., description="Word to look up")):
    """Forward lookup: word → meaning"""
    cache_key = make_cache_key("define", word)
    cached = json_get(cache_key)
    if cached:
        return {"source": "cache", "result": cached}

    word_lower = word.lower()

    if word_lower in trie:
        entry = dictionary_data.get(word, dictionary_data.get(word_lower))
        if entry:
            json_set(cache_key, entry)
            return {"source": "trie", "result":entry}
        
    suggestions = prefix_search(trie, word_lower[:3])
    if suggestions:
        return {
            "error": "Word not found, Did you mean one of these?",
            "suggestions": suggestions[:5]
        }

    return {"error": "Word not found"}

@app.get("/reverse")
def reverse_lookup(meaning: str = Query(..., description="Meaning or phrase to find similar words")):
    """Reverse lookup: meaning → semantically similar words"""
    cache_key = make_cache_key("reverse", meaning)
    cached = json_get(cache_key)
    if cached:
        return {"source": "cache", "result": cached}

    # Embed the query
    query_emb = embed_texts([meaning], model_name=EMBEDDING_MODEL)
    # Search FAISS
    scores, inds = search_index(faiss_index, query_emb[0], top_k=TOP_K)

    results = []
    for score, idx in zip(scores, inds):
        entry = dictionary_data[idx]
        results.append({
            "word": entry["word"],
            "meaning": entry["meaning"],
            "score": float(score)
        })

    # Optional: Rerank results
    results = bm25_rerank(meaning, results, top_k=TOP_K)

    json_set(cache_key, results)
    return {"source": "faiss", "result": results}

@app.get("/autocomplete")
def autocomplete(prefix: str = Query(..., min_length=1)):
    # prefix search should be case-insensitive depending on trie build
    p = prefix.strip().lower()
    matches = prefix_search(trie, p)
    return {"matches": matches[:50]}

@app.get("/", include_in_schema=False)
def root():
    index_file = static_dir / "index.html"
    return index_file.read_text(encoding=utf-8)
