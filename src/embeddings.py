from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

_model = None

def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model = get_model(model_name)
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False
    )
    return embs.astype("float32")
