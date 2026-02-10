import numpy as np

from src import scorer


def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert scorer.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_self():
    a = np.array([1.0, 2.0, 3.0])
    assert scorer.cosine_similarity(a, a) == 1.0
