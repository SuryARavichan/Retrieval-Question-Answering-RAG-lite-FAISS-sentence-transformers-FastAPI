from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedder import Embedder
from index_store import FaissIndexStore
import uvicorn
import numpy as np
import json
from typing import List

app = FastAPI(title="RAG-lite FAISS API")

embedder = Embedder()
dim = embedder.model.get_sentence_embedding_dimension()
store = FaissIndexStore(dim=dim)

class IngestReq(BaseModel):
    id: str
    text: str

class QueryReq(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ingest")
def ingest(doc: IngestReq):
    vec = embedder.encode([doc.text])
    store.add(vec, [{"id": doc.id, "text": doc.text}])
    return {"status": "ok", "id": doc.id}

@app.post("/bulk_ingest")
def bulk_ingest(docs: List[IngestReq]):
    texts = [d.text for d in docs]
    ids = [d.id for d in docs]
    vecs = embedder.encode(texts)
    metas = [{"id": i, "text": t} for i, t in zip(ids, texts)]
    store.add(vecs, metas)
    return {"status": "ok", "ingested": len(docs)}

@app.post("/query")
def query(req: QueryReq):
    qvec = embedder.encode([req.query])
    results = store.search(qvec, top_k=req.top_k)[0]
    # Build a simple "answer" by concatenating top snippets (can call a generator later)
    answer = "\n\n".join([r["meta"]["text"] for r in results])
    return {"query": req.query, "answer": answer, "results": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
