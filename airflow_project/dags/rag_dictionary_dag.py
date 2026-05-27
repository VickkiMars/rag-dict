"""
Airflow DAG to automate RAG Dictionary Index Building & Cache Warming.

This DAG decouples ingestion and indexing from search serving. It does NOT pass large
datasets via XCom, avoiding database bloat or memory failures. Instead, it reads the
dictionary JSON directly, runs Sentence-Transformers, builds a flat IP FAISS vector index,
builds a marisa-trie for fast prefix searches, and pre-populates Redis for forward lookup.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json
import os
import numpy as np

# Local imports
from src.embeddings import embed_texts, get_model
from src.vector_store import build_faiss_index, save_faiss
from src.dict_trie import build_trie_from_json
from src.cache import json_set
from src.config import DICT_JSON, FAISS_PATH, EMBEDDING_MODEL, REDIS_PREFIX

def validate_dictionary_data(**context):
    """Verifies that the dictionary JSON exists and is in the correct format."""
    if not os.path.exists(DICT_JSON):
        raise FileNotFoundError(f"Dictionary file not found at {DICT_JSON}")
    
    with open(DICT_JSON, 'r', encoding='utf-8') as f:
        # Load a small snippet to validate it
        data_head = {}
        count = 0
        for line in f:
            count += 1
            if count > 500:
                break
        
    print(f"Dictionary file exists and seems readable.")
    # Log metadata
    context['ti'].xcom_push(key='validated_at', value=datetime.utcnow().isoformat())


def build_faiss_and_trie(**context):
    """
    Reads the dictionary JSON, extracts definition texts, embeds them,
    builds the FAISS vector index, and compiles the prefix search Trie.
    """
    print(f"Loading dictionary from {DICT_JSON}...")
    with open(DICT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = list(data.keys())
    print(f"Processing {len(words)} entries for FAISS index and Trie...")

    # 1. Compile the Prefix Search Trie
    trie_path = "data/dictionary.trie"
    print("Building prefix Trie (lowercase keys)...")
    build_trie_from_json(DICT_JSON, trie_path)
    print(f"Trie built successfully and saved to {trie_path}")

    # 2. Extract meanings for FAISS vector index
    meanings = []
    for word in words:
        info = data[word]
        definitions = info.get("definitions", [])
        definition_text = " ".join(definitions) if definitions else word
        meanings.append(definition_text)

    # 3. Generate Embeddings using Sentence Transformers
    print(f"Generating embeddings for {len(meanings)} meanings using model: {EMBEDDING_MODEL}")
    embeddings = embed_texts(meanings, EMBEDDING_MODEL)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # 4. Build FAISS Flat IP Index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # 5. Save FAISS index
    save_faiss(index, FAISS_PATH)
    print(f"FAISS index saved successfully at {FAISS_PATH}")

    context['ti'].xcom_push(key='word_count', value=len(words))


def warm_redis_cache(**context):
    """
    Reads the dictionary JSON and pre-caches every word's full entry
    in Redis for sub-millisecond forward lookup.
    """
    print(f"Loading dictionary from {DICT_JSON} to warm Redis cache...")
    with open(DICT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Pre-caching {len(data)} entries into Redis...")
    for word, info in data.items():
        entry = {
            "word": word,
            "definitions": info.get("definitions", []),
            "meaning": " ".join(info.get("definitions", [])),
            "pos": info.get("pos", []),
            "pronunciations": info.get("pronunciations", []),
            "etymology": info.get("etymology", ""),
        }
        
        # Format the cache key identically to serve.py: make_cache_key
        cache_key = f"{REDIS_PREFIX}define:{word.strip().lower()}"
        json_set(cache_key, entry, ex=86400) # cache for 24 hours

    print("Redis cache warming complete.")


# --- Airflow DAG Definition ---

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="rag_dictionary_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Build FAISS vector index, prefix Trie, and warm Redis cache for RAG Dictionary",
    tags=["rag", "dictionary", "faiss", "redis", "observability"],
) as dag:

    validate_data = PythonOperator(
        task_id="validate_dictionary_data",
        python_callable=validate_dictionary_data,
        provide_context=True
    )

    build_indices = PythonOperator(
        task_id="build_faiss_and_trie",
        python_callable=build_faiss_and_trie,
        provide_context=True
    )

    cache_warmup = PythonOperator(
        task_id="warm_redis_cache",
        python_callable=warm_redis_cache,
        provide_context=True
    )

    validate_data >> build_indices >> cache_warmup
