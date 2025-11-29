"""
Jobfit FastAPI Application

Simple server that serves static HTML pages using lxml for parsing.
No templates, no string manipulation - just direct file serving.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(
    title="Jobfit",
    description="Next-generation AI-powered job search and career optimization",
    version="0.1.0"
)

# Path to static files
STATIC_DIR = Path(__file__).parent.parent / "static"

# Mount static files for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Serve styles.css at root level for relative path compatibility
@app.get("/styles.css")
async def styles():
    """Serve main stylesheet"""
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


# HTML Page Routes - Direct file serving
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
    return FileResponse(STATIC_DIR / "dataops.html")


@app.get("/mlops", response_class=HTMLResponse)
@app.get("/mlops.html", response_class=HTMLResponse)
async def mlops():
    """Serve ML operations hub page"""
    return FileResponse(STATIC_DIR / "mlops.html")


@app.get("/designsystem", response_class=HTMLResponse)
@app.get("/designsystem.html", response_class=HTMLResponse)
async def designsystem():
    """Serve design system page"""
    return FileResponse(STATIC_DIR / "designsystem.html")


# API endpoints will be added here later
# For now, keeping it simple - just serving static pages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
