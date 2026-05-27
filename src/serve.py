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
from src.monitor import record_hit, record_miss, record_vector_lookup
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path

static_dir = Path(__file__).resolve().parent / "frontend"

# Initialize FastAPI
app = FastAPI(title="RAG Dictionary API", version="1.0")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Load Data and FAISS Index ---
trie = None
dictionary_data = None
dictionary_keys = None
word_mapping_lower = None
faiss_index = None
model = None
faiss_status = "Not started"

# --- Metrics Setup ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.on_event("startup")
def startup_event():
    global trie, dictionary_data, dictionary_keys, word_mapping_lower
    
    print("Loading dictionary JSON...")
    with open(DICT_JSON, "r", encoding="utf-8") as f:
        dictionary_data = json.load(f)
    
    dictionary_keys = list(dictionary_data.keys())
    word_mapping_lower = {w.lower(): w for w in dictionary_keys}
    print(f"Loaded {len(dictionary_keys)} words into memory.")

    # Load/Build Trie
    trie_path = "data/dictionary.trie"
    if not os.path.exists(trie_path):
        print("Trie file not found, building from JSON...")
        trie = build_trie_from_json(DICT_JSON, trie_path)
    else:
        print("Loading Trie...")
        trie = load_trie(trie_path)

    # Launch background thread to load model and FAISS index
    import threading
    threading.Thread(target=load_model_and_faiss_background, daemon=True).start()
    
    # Warm Redis locally in the background
    threading.Thread(target=warm_redis_locally, daemon=True).start()
    
    print("FastAPI server started successfully! Port 8000 is open.")

def load_model_and_faiss_background():
    global model, faiss_index, faiss_status
    faiss_status = "Loading embedding model..."
    try:
        print("Loading embedding model in background...")
        model = get_model(EMBEDDING_MODEL)
        
        # Load/Build FAISS index
        if not os.path.exists(FAISS_PATH):
            print("FAISS index not found, building initial index in background...")
            faiss_status = "Compiling FAISS vector index (this may take a few minutes)..."
            faiss_index, _ = prepare_data_and_index(EMBEDDING_MODEL)
            print("FAISS index built and saved successfully in background.")
        else:
            print("Loading FAISS index in background...")
            faiss_status = "Loading FAISS index..."
            faiss_index = load_faiss(FAISS_PATH)
            print("FAISS index loaded successfully in background.")
        faiss_status = "Ready"
    except Exception as e:
        print(f"Error during background startup: {e}")
        faiss_status = f"Error: {e}"

def warm_redis_locally():
    from src.cache import r as redis_client
    # Use a check key to see if cache is already warmed
    check_key = make_cache_key("define", "serendipity")
    if redis_client.exists(check_key):
        print("Redis cache is already warmed.")
        return

    print("Warming Redis cache locally in background...")
    try:
        count = 0
        for word, info in dictionary_data.items():
            entry = {
                "word": word,
                "definitions": info.get("definitions", []),
                "meaning": " ".join(info.get("definitions", [])),
                "pos": info.get("pos", []),
                "pronunciations": info.get("pronunciations", []),
                "etymology": info.get("etymology", ""),
            }
            cache_key = make_cache_key("define", word)
            json_set(cache_key, entry, ex=86400)
            count += 1
        print(f"Background Redis cache warming complete. Cached {count} entries.")
    except Exception as e:
        print(f"Error during background Redis cache warming: {e}")

# --- Utility ---
def make_cache_key(prefix: str, query: str) -> str:
    return f"{REDIS_PREFIX}{prefix}:{query.strip().lower()}"

# --- API Endpoints ---

@app.get("/define")
def define(word: str = Query(..., description="Word to look up")):
    """Forward lookup: word ➔ meaning"""
    word_clean = word.strip()
    cache_key = make_cache_key("define", word_clean)
    cached = json_get(cache_key)
    if cached:
        record_hit()
        return {"source": "cache", "result": cached}

    record_miss()
    word_lower = word_clean.lower()
    actual_word = word_mapping_lower.get(word_lower)

    if actual_word:
        entry = dictionary_data[actual_word]
        res_entry = {
            "word": actual_word,
            "definitions": entry.get("definitions", []),
            "meaning": " ".join(entry.get("definitions", [])),
            "pos": entry.get("pos", []),
            "pronunciations": entry.get("pronunciations", []),
            "etymology": entry.get("etymology", ""),
        }
        json_set(cache_key, res_entry)
        return {"source": "database", "result": res_entry}
        
    suggestions_lower = prefix_search(trie, word_lower[:3])
    suggestions = [word_mapping_lower.get(s, s) for s in suggestions_lower if s in word_mapping_lower]
    
    if suggestions:
        return {
            "error": "Word not found. Did you mean one of these?",
            "suggestions": suggestions[:5]
        }

    return {"error": "Word not found"}

@app.get("/reverse")
def reverse_lookup(meaning: str = Query(..., description="Meaning or phrase to find similar words")):
    """Reverse lookup: meaning ➔ semantically similar words"""
    global faiss_index
    if faiss_index is None:
        return {
            "error": "The semantic search index is currently compiling offline in the background. "
                     "Exact word definitions are active! Please try your semantic lookup again in a few minutes.",
            "status": faiss_status
        }
        
    meaning_clean = meaning.strip()
    cache_key = make_cache_key("reverse", meaning_clean)
    cached = json_get(cache_key)
    if cached:
        record_hit()
        return {"source": "cache", "result": cached}

    record_miss()
    # Embed the query
    query_emb = embed_texts([meaning_clean], model_name=EMBEDDING_MODEL)
    record_vector_lookup()
    
    # Search FAISS
    scores, inds = search_index(faiss_index, query_emb[0], top_k=TOP_K)

    retrieved = []
    for score, idx in zip(scores, inds):
        if idx < 0 or idx >= len(dictionary_keys):
            continue
        word = dictionary_keys[idx]
        entry = dictionary_data[word]
        retrieved.append({
            "word": word,
            "meaning": " ".join(entry.get("definitions", [])),
            "score": float(score),
            "pos": entry.get("pos", []),
            "definitions": entry.get("definitions", [])
        })

    # Lexical reranking
    results = bm25_rerank(meaning_clean, retrieved, top_k=TOP_K)

    json_set(cache_key, results)
    return {"source": "faiss", "result": results}

@app.get("/autocomplete")
def autocomplete(prefix: str = Query(..., min_length=1)):
    p = prefix.strip().lower()
    matches_lower = prefix_search(trie, p)
    matches = [word_mapping_lower.get(m, m) for m in matches_lower if m in word_mapping_lower]
    return {"matches": matches[:50]}

@app.get("/", include_in_schema=False)
def root():
    index_file = static_dir / "index.html"
    return index_file.read_text(encoding="utf-8")
