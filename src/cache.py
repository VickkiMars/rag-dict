"""
Redis cache utilities for:
- caching query results
- storing simple dictionary key->value
- caching embeddings (optional)
"""

import redis
import json
from src.config import REDIS_HOST, REDIS_PORT
from typing import Optional
import numpy as np
import base64

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


def kv_set(key: str, value: str, ex: int = 3600):
    r.set(key, value, ex=ex)

def kv_get(key: str) -> Optional[str]:
    val = r.get(key)
    if val is None:
        return None
    try:
        return val.decode("utf-8")
    except Exception:
        return val
    
def json_set(key: str, obj, ex:int = 3600):
    r.set(key, json.dumps(obj, ensure_ascii=False), ex=ex)

def json_get(key: str):
    val = r.get(key)
    if val is None:
        return None
    return json.loads(val)

def emb_set(key: str, arr: np.ndarray, ex: int = 86400):
    b = base64.b64encode(arr.tobytes())
    r.set(key, b, ex=ex)

def emb_get(key: str, dtype=np.float32, shape=None):
    b = r.get(key)
    if b is None:
        return None
    raw = base64.b64decode(b)
    arr = np.frombuffer(raw, dtype=dtype)
    if shape:
        return arr.reshape(shape)
    return arr