import os

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
FAISS_PATH = os.environ.get("FAISS_PATH", "/app/data/embeddings.faiss")
DICT_JSON = os.environ.get("DICT_JSON", "/app/data/dictionary.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5
REDIS_PREFIX = "rag_dict:"
