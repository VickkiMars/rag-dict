import os

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
FAISS_PATH = os.environ.get("FAISS_PATH", "data/embeddings.faiss")
DICT_JSON = os.environ.get("DICT_JSON", "data/dictionary.json")

LOCAL_MODEL_PATH = "/home/kami/Desktop/codebase/rag-dict/models/all-MiniLM-L6-v2"
EMBEDDING_MODEL = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5
REDIS_PREFIX = "rag_dict:"
