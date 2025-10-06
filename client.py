import requests, json
BASE="http://127.0.0.1:8000"
# ingest sample docs
docs = [
    {"id":"d1","text":"Python is an interpreted, high-level programming language."},
    {"id":"d2","text":"FAISS is a library for efficient similarity search and clustering of dense vectors."},
    {"id":"d3","text":"SentenceTransformers provide easy-to-use sentence embeddings."}
]
resp = requests.post(f"{BASE}/bulk_ingest", json=docs)
print(resp.json())
# query
q = {"query":"how to get embeddings for sentences?", "top_k":3}
resp = requests.post(f"{BASE}/query", json=q)
print(json.dumps(resp.json(), indent=2))
