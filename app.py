from pathlib import Path
from typing import Optional

import gradio as gr

from src import parser
from src import scorer


def _load_text(file: Optional[gr.File]) -> str:
    if file is None:
        return ""
    path = Path(file.name)
    if path.suffix.lower() == ".pdf":
        return parser.read_pdf(path)
    return parser.read_text_file(path)


def analyze(resume_file, jd_file, resume_text_input: str, jd_text_input: str):
    resume_text = _load_text(resume_file) if resume_file else resume_text_input
    jd_text = _load_text(jd_file) if jd_file else jd_text_input

    if not resume_text or not jd_text:
        return 0.0, [], [], []

    similarity = scorer.overall_similarity(resume_text, jd_text)
    sections = scorer.rank_resume_sections(resume_text, jd_text)
    missing = scorer.missing_skills(resume_text, jd_text)
    suggestions = scorer.ats_suggestions(resume_text, jd_text)

    sections_display = [
        {"Section": row["title"], "Score": round(row["score"], 3), "Content": row["content"]}
        for row in sections
    ]

    return round(similarity, 3), sections_display, missing, suggestions


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Resume vs JD Matcher") as demo:
        gr.Markdown("## Resume vs Job Description Matcher\nUpload a resume and job description or paste text.")
        with gr.Row():
            with gr.Column():
                resume_file = gr.File(label="Resume (PDF or TXT)", file_types=[".pdf", ".txt"], type="binary")
                resume_text = gr.Textbox(label="Resume Text (optional)", lines=10)
            with gr.Column():
                jd_file = gr.File(label="Job Description (PDF or TXT)", file_types=[".pdf", ".txt"], type="binary")
                jd_text = gr.Textbox(label="Job Description Text (optional)", lines=10)
        analyze_btn = gr.Button("Analyze")

        similarity_out = gr.Number(label="Cosine Similarity", precision=3)
        sections_out = gr.Dataframe(headers=["Section", "Score", "Content"], wrap=True)
        missing_out = gr.HighlightedText(label="Missing Skills", combine_adjacent=True)
        suggestions_out = gr.Markdown(label="ATS Suggestions")

        def _format_missing(tokens):
            return [(token, "missing") for token in tokens]

        def _format_suggestions(lines):
            return "\n".join([f"- {line}" for line in lines])

        analyze_btn.click(
            fn=analyze,
            inputs=[resume_file, jd_file, resume_text, jd_text],
            outputs=[similarity_out, sections_out, missing_out, suggestions_out],
            preprocess=False,
            postprocess=False,
        ).then(_format_missing, missing_out, missing_out).then(_format_suggestions, suggestions_out, suggestions_out)

    return demo


def main():
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
