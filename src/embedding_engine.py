import functools
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# Cache the model to avoid repeated downloads and speed up batch inferences.
@functools.lru_cache(maxsize=1)
def _load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = _load_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
