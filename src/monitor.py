from prometheus_client import Counter

CACHE_HIT = Counter("rag_cache_hit_total", "Total cache hits")
CACHE_MISS = Counter("rag_cache_miss_total", "Total cache misses")
VECTOR_LOOKUP = Counter("rag_vector_lookup_total", "Total vector store lookups")

def record_hit():
    CACHE_HIT.inc()

def record_miss():
    CACHE_MISS.inc()

def record_vector_lookup():
    VECTOR_LOOKUP.inc()