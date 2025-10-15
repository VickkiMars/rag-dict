"""
Airflow DAG to automate RAG Dictionary Index Building

Pipeline steps:
1. Load dictionary data
2. Generate embeddings using SentenceTransformers
3. Build FAISS index and save to disk
4. Cache entries and metadata in Redis
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json
import os

# Local imports
from src.embeddings import embed_texts
from src.vector_store import build_faiss_index, save_faiss
from src.cache import json_set
from src.config import DICT_JSON, FAISS_PATH, EMBEDDING_MODEL, REDIS_PREFIX

# -------------------------------------------------------------------
# --- Helper Functions for Airflow Tasks
# -------------------------------------------------------------------

def load_dictionary():
    """Reads the dictionary JSON file from disk."""
    with open(DICT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from dictionary.")
    return data


def generate_embeddings(**context):
    """
    Generates embeddings for all dictionary meanings and pushes to XCom.
    """
    data = context['ti'].xcom_pull(task_ids='load_dictionary')
    meanings = [entry["meaning"] for entry in data]

    print(f"Generating embeddings for {len(meanings)} meanings...")
    embeddings = embed_texts(meanings, EMBEDDING_MODEL)

    # Save intermediate results to XCom
    context['ti'].xcom_push(key='embeddings', value=embeddings.tolist())
    context['ti'].xcom_push(key='data', value=data)
    print("Embeddings generated and pushed to XCom.")


def build_faiss_and_cache(**context):
    """
    Builds the FAISS index and caches dictionary entries in Redis.
    """
    import numpy as np
    from src.cache import r  # redis client

    embeddings = np.array(context['ti'].xcom_pull(key='embeddings', task_ids='generate_embeddings')).astype('float32')
    data = context['ti'].xcom_pull(key='data', task_ids='generate_embeddings')

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    save_faiss(index, FAISS_PATH)
    print(f"FAISS index saved at {FAISS_PATH}")

    print("Caching entries in Redis...")
    for entry in data:
        key = f"{REDIS_PREFIX}{entry['word']}"
        json_set(key, entry)
    print("All entries cached successfully.")


# -------------------------------------------------------------------
# --- Airflow DAG Definition
# -------------------------------------------------------------------

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
    description="Build FAISS index and Redis cache for RAG Dictionary",
    tags=["rag", "dictionary", "faiss", "redis"],
) as dag:

    load_dictionary_task = PythonOperator(
        task_id="load_dictionary",
        python_callable=load_dictionary
    )

    generate_embeddings_task = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
        provide_context=True
    )

    build_faiss_and_cache_task = PythonOperator(
        task_id="build_faiss_and_cache",
        python_callable=build_faiss_and_cache,
        provide_context=True
    )

    load_dictionary_task >> generate_embeddings_task >> build_faiss_and_cache_task
