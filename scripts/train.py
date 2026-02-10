"""Offline preprocessing script to embed resume/JD pairs and persist vectors."""

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from src import parser
from src.embedding_engine import embed_texts


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_any(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return parser.read_pdf(path)
    return parser.read_text_file(path)


def main(config_path: Path = Path("configs/config.yaml")) -> None:
    config = load_config(config_path)
    model_name = config.get("model", {}).get("name", "sentence-transformers/all-MiniLM-L6-v2")
    raw_dir = Path(config.get("paths", {}).get("raw_data", "data/raw"))
    processed_dir = Path(config.get("paths", {}).get("processed_data", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = raw_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs.csv at {pairs_path}")

    resumes, jds = [], []
    with pairs_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_path = raw_dir / row["resume_path"]
            jd_path = raw_dir / row["jd_path"]
            resumes.append(read_any(resume_path))
            jds.append(read_any(jd_path))

    resume_emb = embed_texts(resumes, model_name=model_name)
    jd_emb = embed_texts(jds, model_name=model_name)

    out_path = processed_dir / "embeddings.npz"
    np.savez(out_path, resume=resume_emb, jd=jd_emb)
    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
