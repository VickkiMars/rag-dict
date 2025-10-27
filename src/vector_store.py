import faiss
import numpy as np
from typing import List, Tuple
import os
import json
from src.config import FAISS_PATH, DICT_JSON, TOP_K
from src.embeddings import embed_texts
from rank_bm25 import BM25Okapi


# -----------------------------
# 1. BUILD FAISS INDEX
# -----------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index for fast similarity search.

    Args:
        embeddings (np.ndarray): 2D array of text embeddings.

    Returns:
        faiss.Index: A FAISS index ready for searching.
    """
    dim = embeddings.shape[1]                   # get vector dimensionality
    index = faiss.IndexFlatIP(dim)              # create index using inner product (cosine similarity)
    faiss.normalize_L2(embeddings)              # normalize all embedding vectors to unit length
    index.add(embeddings)                       # store all vectors in the index
    return index

# -----------------------------
# 2. SAVE FAISS INDEX
# -----------------------------
def save_faiss(index: faiss.Index, path: str = FAISS_PATH):
    """
    Saves a FAISS index to disk.

    Args:
        index (faiss.Index): The FAISS index to save.
        path (str): File path for saving the index.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure directory exists
    faiss.write_index(index, path)                     # serialize and save index to file

# -----------------------------
# 3. LOAD FAISS INDEX
# -----------------------------
def load_faiss(path: str = FAISS_PATH) -> faiss.Index:
    """
    Loads a saved FAISS index from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    index = faiss.read_index(path)
    return index

# -----------------------------
# 4. PREPARE DATA AND BUILD INDEX
# -----------------------------
def prepare_data_and_index(model_name: str) -> Tuple[faiss.Index, list]:
    """
    Loads dictionary data, embeds definitions, builds FAISS index, and saves it.
    """
    with open(DICT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten entries into list form
    entries = []
    meanings = []

    for word, info in data.items():
        definition_text = " ".join(info.get("definitions", []))
        meanings.append(definition_text)
        entries.append({"word": word, **info})

    embeddings = embed_texts(meanings, model_name=model_name)
    index = build_faiss_index(embeddings)
    save_faiss(index)

    return index, entries

# -----------------------------
# 5. BM25 RERANKER
# -----------------------------
def rerank_bm25(query: str, docs: List[str]) -> List[str]:
    """
    Uses BM25 to re-rank semantic results based on lexical similarity.

    Args:
        query (str): The user query.
        docs (List[str]): Candidate document texts.

    Returns:
        List[str]: Re-ranked documents sorted by BM25 score.
    """
    tokenized_docs = [d.split() for d in docs]            # tokenize documents
    bm25 = BM25Okapi(tokenized_docs)                      # initialize BM25 with tokenized corpus
    scores = bm25.get_scores(query.split())               # score each doc by lexical overlap
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

# -----------------------------
# 6. SEARCH INDEX
# -----------------------------
def search_index(index: faiss.Index, query_emb: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs FAISS similarity search.
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)
    scores, inds = index.search(query_emb, top_k)
    return scores[0], inds[0]


# -----------------------------
# 7. SEARCH MEANING
# -----------------------------
def search_meaning(query: str, index, data, model_name: str, top_k: int = TOP_K):
    """
    Given a query, retrieves top-k similar meanings from FAISS and dictionary.
    """
    query_emb = embed_texts([query], model_name=model_name)      # embed the query
    scores, inds = search_index(index, query_emb[0], top_k)      # retrieve top-k matches

    results = []
    for score, idx in zip(scores, inds):
        word = data[idx]["word"]
        meaning = data[idx]["meaning"]
        results.append({
            "word": word,
            "meaning": meaning,
            "score": float(score)
        })
    return results
