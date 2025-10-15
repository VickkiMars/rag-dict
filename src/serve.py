# src/serve.py

from fastapi import FastAPI, Query
from src.cache import kv_get, kv_set, json_get, json_set
from src.embeddings import embed_texts
from src.vector_store import load_faiss_index, search_index
from src.reranker import bm25_rerank
from src.config import (
    DICT_JSON, FAISS_PATH, EMBEDDING_MODEL, TOP_K, REDIS_PREFIX
)
from prometheus_fastapi_instrumentator import Instrumentator
import json
import numpy as np
import os

# Initialize FastAPI
app = FastAPI(title="RAG Dictionary API", version="1.0")

# --- Load Data and FAISS Index ---
with open(DICT_JSON, "r", encoding="utf-8") as f:
    dictionary_data = json.load(f)

faiss_index = load_faiss_index(FAISS_PATH)

# --- Metrics Setup ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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

    # search local dictionary data
    for item in dictionary_data:
        if item["word"].lower() == word.lower():
            json_set(cache_key, item)
            return {"source": "dict", "result": item}

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

@app.get("/")
def root():
    return {"message": "Welcome to the Local RAG Dictionary API!"}
