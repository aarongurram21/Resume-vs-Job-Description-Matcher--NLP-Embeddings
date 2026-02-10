"""Compute cosine similarities for embedded resume/JD pairs."""

from pathlib import Path
import numpy as np

from src.scorer import cosine_similarity


def main(embeddings_path: Path = Path("data/processed/embeddings.npz")) -> None:
    if not embeddings_path.exists():
        raise FileNotFoundError("Run scripts/train.py first to generate embeddings.npz")

    data = np.load(embeddings_path)
    resumes = data["resume"]
    jds = data["jd"]

    if resumes.shape[0] != jds.shape[0]:
        raise ValueError("Embedding arrays are misaligned")

    scores = [cosine_similarity(resumes[i], jds[i]) for i in range(resumes.shape[0])]
    for idx, score in enumerate(scores, start=1):
        print(f"Pair {idx}: cosine similarity={score:.3f}")


if __name__ == "__main__":
    main()
