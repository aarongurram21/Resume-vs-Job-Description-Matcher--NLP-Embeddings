# Resume vs Job Description Matcher

Lightweight NLP app that compares a resume to a job description using Sentence-Transformers embeddings. It computes cosine similarity, surfaces missing skills, ranks resume sections, and offers ATS-friendly suggestions via a Gradio UI.

## Features
- PDF or text ingestion for both resume and job description
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity scoring
- Missing skill highlighting (naive keyword extraction)
- Ranked resume sections against the JD
- ATS-friendly phrasing suggestions
- Sample data and a unit test for the similarity function

## Project Structure (classic ML-style)
```
app.py                  # Gradio UI for resume vs JD
main.py                 # CLI entrypoint (train/evaluate classic pipeline)
src/
  embedding_engine.py   # Embedding helper
  parser.py             # PDF/text parsing and skill extraction
  scorer.py             # Similarity, ranking, suggestions
  data/load_data.py     # Raw/processed IO helpers
  features/build_features.py # Simple preprocessing pipeline
  models/model.py       # Model factory
  train.py              # Classic train routine
  evaluate.py           # Classic eval routine
scripts/                # Offline embedding pipeline (resume/JD pairs)
data/raw                # Raw samples (resume/JD and toy CSV)
data/processed          # Processed splits/embeddings
models/                 # Saved models
reports/                # Evaluation reports
notebooks/              # EDA and scaffold notebook
sample_data/            # Example resume and JD (UI demo)
tests/                  # Pytest suite
requirements.txt        # Dependencies
Makefile                # Common tasks
.vscode/                # Tasks/launch configs
Dockerfile, .dockerignore
```

## Setup
1. Python 3.10+ recommended.
2. Install dependencies (CPU-only):
   ```bash
   pip install -r requirements.txt
   ```

## Run the app
```bash
python app.py
```
Then open the Gradio link printed in the console.

## Usage
- Upload a resume (PDF/TXT) and a JD (PDF/TXT) or paste text directly.
- Click **Analyze** to see:
  - Cosine similarity (overall match)
  - Ranked resume sections
  - Missing skills list
  - ATS-friendly suggestions

### Classic ML pipeline (toy data)
- Train: `python main.py train-cmd`
- Evaluate: `python main.py evaluate-cmd`
- Notebook scaffold: see `notebooks/classic_ml_scaffold.ipynb` for end-to-end setup, Makefile generation, and VS Code task generation.

## Evaluation Example
Using the provided samples:
- Resume: `sample_data/resume.txt`
- JD: `sample_data/job_description.txt`

Run:
```bash
python app.py
```
Then load the sample texts; you should see:
- Cosine similarity: around 0.6–0.7 (illustrative, depends on hardware/model)
- Top sections include Experience and Skills with higher scores
- Missing skills might include any JD phrases absent from the resume wording

## Testing
```bash
pytest tests/test_scorer.py
```

## Notes
- Missing skills use a simple keyword heuristic; for production, consider a proper skill taxonomy and phrase normalization.
- For larger deployments, cache embeddings and add rate limiting around model calls.
