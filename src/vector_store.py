import faiss
import numpy as np
from typing import List, Tuple
import os
from src.config import FAISS_PATH, FAISS_INDEX_DIM
from rank_bm25 import BM250kapi

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_faiss(index: faiss.Index, path: str = FAISS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss(path: str = FAISS_PATH) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
        return none
    index = faiss.read_index(path)
    return index

def prepare_data_and_index(model_name: str) -> Tuple[faiss.Index, dict]:
    with open(DICT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meanings = [entry["meaning"] for entry in data]
    embeddings = embed_texts(meanings, model_name=model_name)
    index - build_faiss_index(embeddings)
    save_faiss(index)
    return index, data

def rerank_bm25(query:str, docs: List[str]) -> List[str]:
    bm25 = BM250kapi([d.split() for d in docs])
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

def search_index(index: faiss.Index, query_emb: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        index (faiss.Index): _description_
        query_emb (np.ndarray): shape (d,) or (1,d)
        top_k (int, optional): _description_. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (scores, indices)
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)
    scores, inds = index.search(query_emb, top_k)
    return scores[0], inds[0]


def search_meaning(query: str, index, data, model_name: str, top_k: int = TOP_K):
    query_emb = embed_texts([query], model_name=model_name)
    scores, inds = search_index(index, query_emb[0], top_k)   # <- call here!

    results = []
    for score, idx in zip(scores, inds):
        word = data[idx]["word"]
        meaning = data[idx]["meaning"]
        results.append({"word": word, "meaning": meaning, "score": float(score)})

    return results