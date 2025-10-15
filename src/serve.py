"""
serve.py

FastAPI service exposing:
- /define?word=WORD (exact forward lookup)
- /reverse?meaning=TEXT (semantic reverse lookup via FAISS)
- /metrics (prometheus)
"""
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

app = FastAPI(
	title="RAG Dictionary API",
	version="1.0"
)

with open(DICT_JSON, "r", encoding="utf-8") as f:
	dictionary_data = json.load(f)

load_faiss_index = load_faiss_index(FAISS_PATH)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

def make_cache_key(prefix: str, query: str) -> str:
	return f"{REDIS_PREFIX}{prefix}:{query.strip().lower()}"

def define(word: str = Query(..., description="Word to look up")):
	"""Forward lookup: word -> meaning"""
	cache_key = make_cache_key("define", word)
	cached = json_get(cache_key)
	if cached:
		return {"source": "cache", "result": cached}

	for item in dictionary_data:
		if item["word"].lower() == word.lower():
			json_set(cache_key, item)
			return {"source": "dict", "result": item}

	return {"error": "Word not found"}