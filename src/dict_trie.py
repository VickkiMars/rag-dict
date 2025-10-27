import marisa_trie
import json
import os

def build_trie_from_json(json_path: str, trie_path: str = "data/dictionary.trie"):
    """Builds a trie from dictionary JSON where top-level keys are words."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    words = list(data.keys())
    trie = marisa_trie.Trie(words)
    trie.save(trie_path)
    print(f"Trie built with {len(words)} entries and saved to {trie_path}")
    return trie

def load_trie(trie_path: str = "data/dictionary.trie"):
    """Loads a previously saved Marisa Trie."""
    return marisa_trie.Trie().load(trie_path)

def prefix_search(trie: marisa_trie.Trie, prefix: str):
    """Returns all dictionary words that start with a given prefix."""
    return trie.keys(prefix)

if __name__ == "__main__":
    build_trie_from_json("/home/kami/Desktop/codebase/rag-dict/cleaned_output.json")
    trie = load_trie()
    matches = prefix_search(trie, "Heinz")
    print(matches)