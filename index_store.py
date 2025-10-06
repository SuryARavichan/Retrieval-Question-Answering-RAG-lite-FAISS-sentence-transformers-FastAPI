import faiss
import pandas as pd
import numpy as np
import os
import json


class FaissIndexStore:
    def __init__(self, dim, index_path="faiss.index", meta_path="meta.json"):
        self.dim = dim
        self.index_path = index_path
        self.math_path = math_path
        self.meta = []
        self_init_index()

    def _init_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.meta = []
    
    def add(self, vectors, metadata):
         assert vectors.shape[1] == self.dim
        # normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(metas)
        self.save()

    def search(self, vector, top_k=5):
        faiss.normalize_L2(vector)
        D, I = self.index.search(vector, top_k)
        results = []
        for idx_list, dist_list in zip(I, D):
            row = []
            for idx, dist in zip(idx_list, dist_list):
                if idx == -1 or idx >= len(self.meta):
                    continue
                row.append({"meta": self.meta[idx], "score": float(dist)})
            results.append(row)
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)