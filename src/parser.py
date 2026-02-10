from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pdfplumber


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return _clean_text("\n".join(pages))


def read_text_file(path: Path) -> str:
    return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def load_resume_and_jd(
    resume_path: Optional[Path] = None,
    resume_text: str = "",
    jd_path: Optional[Path] = None,
    jd_text: str = "",
) -> Tuple[str, str]:
    """Load resume and job description text from files or direct input."""
    if resume_path:
        if resume_path.suffix.lower() == ".pdf":
            resume_text = read_pdf(resume_path)
        else:
            resume_text = read_text_file(resume_path)
    if jd_path:
        if jd_path.suffix.lower() == ".pdf":
            jd_text = read_pdf(jd_path)
        else:
            jd_text = read_text_file(jd_path)

    return _clean_text(resume_text), _clean_text(jd_text)


def simple_skill_extraction(text: str) -> set[str]:
    """Naive skill extraction by capturing capitalized tokens and common separators."""
    # Split on commas, slashes, semicolons, and newlines, then filter words that look skill-like.
    raw_tokens = re.split(r"[,/;\n]", text)
    skills = set()
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        # Keep short phrases (1-3 words) that contain letters or digits.
        if 1 <= len(token.split()) <= 4 and re.search(r"[A-Za-z0-9]", token):
            skills.add(token.lower())
    return skills
