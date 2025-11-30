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
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
from docx import Document

# Add src directory to path for uniform imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from .dataops_renderer import render_dataops_page
from .mlops_renderer import render_mlops_page
from workers.embeddings.search import Search
from shared.config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="Jobfit",
    description="Next-generation AI-powered job search and career optimization",
    version="0.1.0"
)

# Path to static files
STATIC_DIR = Path(__file__).parent.parent / "static"
DATAOPS_TEMPLATE = STATIC_DIR / "dataops.html"
MLOPS_TEMPLATE = STATIC_DIR / "mlops.html"
SEARCH_HISTORY_LOG = get_settings().models_dir / "embeddings" / "metrics" / "search_history.jsonl"


# Initialize search object
search_client = Search()

# Mount static files for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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

def extract_text_from_file(file: UploadFile):
    """Extracts text from PDF or DOCX file."""
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        return "".join(page.extract_text() for page in pdf_reader.pages)
    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX.")


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
    """API endpoint for keyword search"""
    start_time = time.time()
    query_params = {"keywords": keywords, "location": location, "skills": skills}
    
    try:
        data = search_client.keyword_search(
            query=keywords,
            location=location,
            skills=skills,
            page=page,
            page_size=page_size
        )
        duration = time.time() - start_time
        log_search_event("keyword", duration, query_params, data['total_results'])
        return data
    except Exception as e:
        duration = time.time() - start_time
        log_search_event("keyword", duration, query_params, 0, status="error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/semantic")
async def api_semantic_search(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """API endpoint for semantic CV search"""
    start_time = time.time()
    query_params = {"filename": file.filename}

    try:
        query_text = extract_text_from_file(file)
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        data = search_client.semantic_search(
            query_text=query_text,
            page=page,
            page_size=page_size
        )
        duration = time.time() - start_time
        log_search_event("semantic", duration, query_params, data['total_results'])
        return data
    except HTTPException as e:
         raise e
    except Exception as e:
        duration = time.time() - start_time
        log_search_event("semantic", duration, query_params, 0, status="error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
