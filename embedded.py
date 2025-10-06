from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        # returns numpy array (n, dim)
        return np.array(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True))
