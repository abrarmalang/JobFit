"""
Jobfit FastAPI Application

Simple server that serves static HTML pages using lxml for parsing.
No templates, no string manipulation - just direct file serving.
"""
import json
import time
import datetime
import os
import sys
import logging
import platform
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
from docx import Document
from functools import lru_cache
from typing import Optional

from fastapi.responses import HTMLResponse, JSONResponse
# COMMENTED OUT: RecommendationEngine requires sentence-transformers (~300MB+ RAM)
# Exceeds Render free tier 512MB limit. Using Groq API for CV matching instead.
# from .recommendation_engine import RecommendationEngine


# Add src directory to path for uniform imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from .dataops_renderer import render_dataops_page
from .mlops_renderer import render_mlops_page
from .cv_loader import build_cv_context
from .cv_renderer import render_cv_review_page
from workers.embeddings.search import Search
from shared.config import get_settings

try:
    import resource  # Unix
except ImportError:  # pragma: no cover
    resource = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

logging.basicConfig(level=logging.INFO)

# Memory and parsing limits
MAX_UPLOAD_SIZE_MB = 5
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_PDF_PAGES = 20
MAX_EXTRACTED_CHARS = 20000
MAX_DESCRIPTION_LENGTH = 600

# --- Memory helpers ---

def _current_memory_mb():
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024

    return float("nan")


def log_memory(stage: str):
    mem = _current_memory_mb()
    message = f"[Jobfit] {stage} memory usage: {mem:.2f} MB"
    print(message)
    logging.getLogger("jobfit.memory").info(message)

# Initialize FastAPI app
app = FastAPI(
    title="Jobfit",
    description="Next-generation AI-powered job search and career optimization",
    version="0.1.0"
)

# COMMENTED OUT: RecommendationEngine uses sentence-transformers (~300MB+ RAM)
# Exceeds Render free tier 512MB limit. Using Groq API for CV matching instead.
#
# @lru_cache(maxsize=1)
# def get_recommendation_engine() -> RecommendationEngine:
#     """
#     Lazily initialize and cache the RecommendationEngine.
#     The engine loads:
#       - job_cluster_model.pkl
#       - job_clusters.parquet
#     from models/trained/
#     """
#     try:
#         return RecommendationEngine()
#     except FileNotFoundError as e:
#         logging.error(f"Model files not found: {e}")
#         logging.error("Please run the training pipeline to generate model files")
#         raise HTTPException(
#             status_code=503,
#             detail="Recommendation engine not available. Model files not found. Please run the training pipeline."
#         )
#
#
# @app.post("/api/recommend", response_class=JSONResponse)
# async def recommend_jobs(
#     cv_text: str = Body(..., embed=True),
#     top_n: Optional[int] = Query(10, ge=1, le=50),
# ):
#     """
#     Recommend jobs for the given CV text.
#     - cv_text: raw CV text from the user
#     - top_n: how many jobs to return (dynamic, default = 10, max = 50)
#     """
#     try:
#         engine = get_recommendation_engine()
#         results = engine.recommend(cv_text=cv_text, top_n=top_n)
#         return {"results": results, "count": len(results), "top_n": top_n}
#     except Exception as e:
#         logging.exception("Error during job recommendation")
#         raise HTTPException(status_code=500, detail=str(e))


# Path to static files
STATIC_DIR = Path(__file__).parent.parent / "static"
DATAOPS_TEMPLATE = STATIC_DIR / "dataops.html"
MLOPS_TEMPLATE = STATIC_DIR / "mlops.html"
CVREVIEW_TEMPLATE = STATIC_DIR / "cvreview.html"
SEARCH_HISTORY_LOG = get_settings().models_dir / "metrics" / "search_history.jsonl"


# Mount static files for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_search_client() -> Search:
    """Lazily instantiate the Search client so heavy indexes load once."""
    client = Search()
    log_memory("Search client initialized")
    return client


@app.on_event("startup")
async def log_startup_memory():
    """Emit process memory usage once the server is ready."""
    log_memory("FastAPI startup")


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


# --- Utility Functions ---

def log_search_event(search_type: str, duration: float, query_params: dict, results_count: int, status: str = "success"):
    """Logs a search event to a JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "search_type": search_type,
        "duration_ms": round(duration * 1000, 2),
        "query": query_params,
        "results_count": results_count,
        "status": status
    }
    with open(SEARCH_HISTORY_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def ensure_file_within_limit(upload: UploadFile):
    """Validate upload size to avoid runaway memory usage."""
    try:
        upload.file.seek(0, os.SEEK_END)
        size = upload.file.tell()
        upload.file.seek(0)
    except Exception:
        # Some SpooledTemporaryFile objects may not support seek/tell once rolled to disk.
        return

    if size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
        )


def _truncate_chunks(chunks):
    """Combine text chunks until the global character budget is hit."""
    total = 0
    collected = []
    for chunk in chunks:
        if not chunk:
            continue
        remaining = MAX_EXTRACTED_CHARS - total
        if remaining <= 0:
            break
        collected.append(chunk[:remaining])
        total += min(len(chunk), remaining)
    return "\n\n".join(collected)


def extract_text_from_file(file: UploadFile):
    """Extract text from PDF/DOCX incrementally with safety limits."""
    ensure_file_within_limit(file)
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        file.file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file.file)

        def pdf_chunks():
            for idx, page in enumerate(pdf_reader.pages):
                if idx >= MAX_PDF_PAGES:
                    break
                yield page.extract_text() or ""

        return _truncate_chunks(pdf_chunks())

    if filename.endswith(".docx"):
        file.file.seek(0)
        doc = Document(file.file)
        return _truncate_chunks(para.text for para in doc.paragraphs)

    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX.")


def sanitize_search_payload(payload: dict) -> dict:
    """Trim large fields in the search payload before sending to clients."""
    if not isinstance(payload, dict):
        return payload

    results = payload.get("results") or []
    for job in results:
        if not isinstance(job, dict):
            continue
        description = job.get("description")
        if isinstance(description, str) and len(description) > MAX_DESCRIPTION_LENGTH:
            job["description"] = description[:MAX_DESCRIPTION_LENGTH].rstrip() + "..."
        full_desc = job.get("full_description")
        if isinstance(full_desc, str) and len(full_desc) > MAX_DESCRIPTION_LENGTH:
            job["full_description"] = full_desc[:MAX_DESCRIPTION_LENGTH].rstrip() + "..."
    return payload


# --- Static Page Routes ---

@app.get("/styles.css")
async def styles():
    """Serve main stylesheet"""
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve landing page"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/jobs", response_class=HTMLResponse)
@app.get("/jobs.html", response_class=HTMLResponse)
async def jobs():
    """Serve job search page"""
    return FileResponse(STATIC_DIR / "jobs.html")


@app.get("/cvreview", response_class=HTMLResponse)
@app.get("/cvreview.html", response_class=HTMLResponse)
async def cvreview():
    """Serve CV review page"""
    return FileResponse(STATIC_DIR / "cvreview.html")


@app.get("/trends", response_class=HTMLResponse)
@app.get("/trends.html", response_class=HTMLResponse)
async def trends():
    """Serve job trends page"""
    return FileResponse(STATIC_DIR / "trends.html")


@app.get("/dataops", response_class=HTMLResponse)
@app.get("/dataops.html", response_class=HTMLResponse)
async def dataops():
    """Serve data operations hub page"""
    html = render_dataops_page(DATAOPS_TEMPLATE)
    return HTMLResponse(content=html)


@app.get("/mlops", response_class=HTMLResponse)
@app.get("/mlops.html", response_class=HTMLResponse)
async def mlops():
    """Serve ML operations hub page with live metrics"""
    html = render_mlops_page(MLOPS_TEMPLATE)
    return HTMLResponse(content=html)


@app.get("/designsystem", response_class=HTMLResponse)
@app.get("/designsystem.html", response_class=HTMLResponse)
async def designsystem():
    """Serve design system page"""
    return FileResponse(STATIC_DIR / "designsystem.html")


# --- API Endpoints ---

@app.get("/api/search/keyword")
async def api_keyword_search(
    keywords: str = Query(None),
    location: str = Query(None),
    skills: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """
    API endpoint for job search using Groq API.

    TEMPORARY: Using simple keyword filtering until Groq integration is complete.
    Previously used RecommendationEngine but it requires sentence-transformers (~300MB+ RAM)
    which exceeds Render free tier 512MB limit.
    """
    start_time = time.time()
    query_params = {"keywords": keywords, "location": location, "skills": skills}

    # If nothing entered, return empty instead of "all jobs"
    if not (keywords or location or skills):
        duration = time.time() - start_time
        log_search_event("keyword", duration, query_params, 0, status="success")
        return {"results": [], "total_results": 0}

    # TODO: Implement Groq-based job matching
    # For now, return a message indicating the feature is being updated
    duration = time.time() - start_time
    log_search_event("keyword", duration, query_params, 0, status="disabled")
    raise HTTPException(
        status_code=503,
        detail="Job search temporarily disabled. Switching to Groq API to reduce memory usage. Please check back soon!"
    )



@app.post("/api/search/semantic")
async def api_semantic_search(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """API endpoint for semantic CV search"""
    start_time = time.time()
    query_params = {"filename": file.filename}

    search_client = get_search_client()

    if not getattr(search_client, "semantic_available", True):
        raise HTTPException(
            status_code=503,
            detail="Semantic search is not enabled on this deployment."
        )

    try:
        query_text = extract_text_from_file(file)
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        data = search_client.semantic_search(
            query_text=query_text,
            page=page,
            page_size=page_size
        )
        data = sanitize_search_payload(data)
        duration = time.time() - start_time
        log_search_event("semantic", duration, query_params, data['total_results'])
        return data
    except HTTPException as e:
         raise e
    except Exception as e:
        duration = time.time() - start_time
        log_search_event("semantic", duration, query_params, 0, status="error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/semantic/text")
async def api_semantic_search_text(
    cv_text: str = Body(..., embed=True),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """API endpoint for semantic CV search using raw text (no file upload)."""
    start_time = time.time()
    query_params = {"source": "cvreview"}

    search_client = get_search_client()

    if not getattr(search_client, "semantic_available", True):
        raise HTTPException(
            status_code=503,
            detail="Semantic search is not enabled on this deployment."
        )

    if not cv_text or not cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text is required for semantic search.")

    try:
        data = search_client.semantic_search(
            query_text=cv_text,
            page=page,
            page_size=page_size
        )
        data = sanitize_search_payload(data)
        duration = time.time() - start_time
        log_search_event("semantic_text", duration, query_params, data['total_results'])
        return data
    except HTTPException as e:
        raise e
    except Exception as e:
        duration = time.time() - start_time
        log_search_event("semantic_text", duration, query_params, 0, status="error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cv/analyze", response_class=HTMLResponse)
async def api_cv_analyze(
    file: UploadFile = File(...),
    target_role: str = Query(None, description="Optional target job role for gap analysis")
):
    """
    API endpoint for CV analysis.
    Returns rendered HTML with analysis results and visualizations.
    """
    try:
        # Extract text from uploaded CV
        cv_text = extract_text_from_file(file)
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        # Build context with analysis and visualizations
        context = build_cv_context(cv_text, target_role)

        # Render HTML with results
        html = render_cv_review_page(CVREVIEW_TEMPLATE, context)

        return HTMLResponse(content=html)

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"CV analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing CV: {str(e)}")


@app.get("/config", response_class=HTMLResponse)
@app.get("/config.html", response_class=HTMLResponse)
async def config():
    """Serve LLM configuration page"""
    return FileResponse(STATIC_DIR / "config.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
