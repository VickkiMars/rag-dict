from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import numpy as np

import sys
sys.path.append('/opt/airflow/src')

from embeddings import embed_texts, save_embeddings, get_model
from vector_store import build_faiss_index, save_index, persist_meta
from cache import store_dict_entry, get_redis
from config import EMBEDDING_MODEL, FAISS_PATH