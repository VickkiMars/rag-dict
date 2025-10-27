from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np


def bm25_rerank(query: str, retrieved_items: List[Dict], top_k: int = 5) -> List[Dict]:
	"""Rerank FAISS results based on BM25 lexical relevance"""
	corpus = [item["meaning"] for item in retrieved_items]
	tokenized_corpus = [doc.lower().split() for doc in corpus]
	tokenized_query = query.lower().split()

	bm25 = BM25Okapi(tokenized_corpus)
	bm25_scores = bm25.get_scores(tokenized_query)

	semantic_scores = np.array([item["score"] for item in retrieved_items])
	semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)
	bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)


	combined_scores = 0.6 * semantic_scores + 0.4 * bm25_scores

	ranked = sorted([
		{**item,"bm25_score": float(bm25_s),"combined_scores": float(cs)}for item, bm25_s, cs in zip(retrieved_items, bm25_scores, combined_scores)],key=lambda x:x["combined_score"], reverse=True)

	return ranked[:top_k]