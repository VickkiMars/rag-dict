import marisa_trie
import json
from src.config import DICT_JSON

def build_trie_from_json(json_path: str = DICT_JSON) -> marisa_trie.Trie:
    """Builds a Marisa-Trie from dictionary JSON data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = [entry["word"] for entry in data]
    trie = marisa_trie.Trie(words)
    trie.save(json_path.replace(".json", ".trie"))
    return trie

def load_trie(trie_path: str = DICT_JSON.replace(".json", ".trie")) -> marisa_trie.Trie:
    """Loads an existing Marisa-Trie from disk."""
    return marisa_trie.Trie().load(trie_path)

def prefix_search(trie: marisa_trie.Trie, prefix: str, limit: int = 10):
    """Returns a list of words starting with the given prefix."""
    return trie.keys(prefix)[:limit]
