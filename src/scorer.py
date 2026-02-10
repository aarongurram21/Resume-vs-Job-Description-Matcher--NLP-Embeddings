from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

from .embedding_engine import embed_texts
from .parser import simple_skill_extraction


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def overall_similarity(resume_text: str, jd_text: str) -> float:
    embeddings = embed_texts([resume_text, jd_text])
    if embeddings.shape[0] < 2:
        return 0.0
    return cosine_similarity(embeddings[0], embeddings[1])


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Return (title, body) tuples for sections inferred from headings."""
    pattern = re.compile(r"(?m)^\s*(\w[\w\s]{1,40})\s*:\s*$")
    sections: List[Tuple[str, str]] = []
    matches = list(pattern.finditer(text))
    if not matches:
        # Fallback: use paragraph chunks.
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        for idx, para in enumerate(paragraphs, start=1):
            sections.append((f"Section {idx}", para))
        return sections

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def rank_resume_sections(resume_text: str, jd_text: str, top_k: int = 5) -> List[Dict[str, object]]:
    jd_emb = embed_texts([jd_text])[0]
    sections = split_sections(resume_text)
    rows: List[Dict[str, object]] = []
    for title, body in sections:
        emb = embed_texts([body])[0]
        score = cosine_similarity(emb, jd_emb)
        rows.append({"title": title, "score": score, "content": body})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:top_k]


def missing_skills(resume_text: str, jd_text: str) -> List[str]:
    resume_skills = simple_skill_extraction(resume_text)
    jd_skills = simple_skill_extraction(jd_text)
    missing = sorted(jd_skills - resume_skills)
    return missing


def ats_suggestions(resume_text: str, jd_text: str) -> List[str]:
    missing = missing_skills(resume_text, jd_text)
    suggestions: List[str] = []
    if missing:
        suggestions.append(
            f"Add explicit mentions of these skills to pass keyword filters: {', '.join(missing[:10])}"
        )
    suggestions.append("Mirror phrasing from the job description for core responsibilities.")
    suggestions.append("Quantify impact in experience bullets (numbers, percentages, time saved).")
    suggestions.append("Use consistent tense and clean section headings for ATS parsing.")
    return suggestions
