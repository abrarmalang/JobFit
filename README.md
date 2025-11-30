# Jobfit Quickstart

The project ships with a FastAPI web app and a small set of Python workers that fetch, process, and embed job data. Follow the steps below to run everything locally.

## 1. Requirements

- Python 3.11 (recommended)
- `pip` and `virtualenv`
- An `.env` file in the project root containing your Adzuna API keys and any optional LLM keys (copy `.env.example` if available or create it manually with `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`)

## 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate               # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt         # FastAPI app + shared libs
pip install -r requirements-worker.txt  # Worker-only deps (SentenceTransformers, etc.)
```

## 3. Run the Web App

From the project root:

```bash
uvicorn webapp.app.main:app --reload
```

The app exposes HTML pages and JSON APIs at `http://127.0.0.1:8000/`.

## 4. Run Backend Workers

Each worker is a simple script; run the ones you need in separate shells (ensure the virtualenv stays activated).

```bash
# Fetch fresh jobs from Adzuna (requires ADZUNA_APP_ID / ADZUNA_APP_KEY)
python scripts/run_collection_adzuna.py

# Clean + consolidate collected jobs into Parquet
python scripts/run_processing.py

# Generate sentence-transformer embeddings for semantic job search
python scripts/run_generate_embeddings.py
```

Additional collectors (remote jobs, diverse search profiles, etc.) live under `scripts/run_collection_*.py` and can be started the same way.

## 5. Troubleshooting

- Missing API keys → check `.env`.
- Large model downloads → the embedding worker pulls `sentence-transformers/all-mpnet-base-v2` on the first run; allow a few minutes.
- To inspect worker health metrics, open `/mlops.html` after the app is running.
