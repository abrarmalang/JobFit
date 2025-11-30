"""
ML Ops Context Loader

Reads ML/AI pipeline metrics and shapes them into display-friendly structures
for the SSR ML Ops page.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_METRICS_DIR = MODELS_DIR / "embeddings" / "metrics"
SEARCH_HISTORY_LOG = EMBEDDING_METRICS_DIR / 'search_history.jsonl'


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    return default


def _read_jsonl(path: Path, limit: int = 5) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    entries: List[Dict[str, Any]] = []
    for raw in lines[-limit:]:
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    # newest first
    return list(reversed(entries))

def _read_all_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def _parse_timestamp(ts: str) -> datetime:
    try:
        if ts and ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.utcnow()


def _format_timestamp(ts: str) -> str:
    if not ts:
        return "N/A"
    dt = _parse_timestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_duration(seconds: float) -> str:
    if not seconds:
        return "0s"
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes == 0:
        return f"{secs}s"
    return f"{minutes}m {secs:02d}s"

def _format_ms(ms: float) -> str:
    if not ms:
        return "0ms"
    if ms < 1000:
        return f"{int(ms)}ms"
    return f"{ms/1000:.2f}s"


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_float(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percentage(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _status_badge_class(status: str) -> str:
    status = (status or "").upper()
    mapping = {
        "RUNNING": "status-running",
        "IDLE": "status-idle",
        "ERROR": "status-error",
        "ONLINE": "status-online",
        "OFFLINE": "status-offline",
        "COMPLETED": "status-completed",
        "SUCCESS": "status-completed",
    }
    return mapping.get(status, "status-idle")


def _status_label(status: str) -> str:
    status = (status or "").upper()
    mapping = {
        "RUNNING": "Running",
        "IDLE": "Idle",
        "ERROR": "Error",
        "ONLINE": "Online",
        "OFFLINE": "Offline",
        "SUCCESS": "Success",
        "COMPLETED": "Completed",
    }
    return mapping.get(status, status.title() if status else "Unknown")


def build_mlops_context() -> Dict[str, Any]:
    """Return ML/AI data ready for SSR rendering."""
    # Read embedding model status and metrics from models/embeddings/metrics/
    model_status = _load_json(EMBEDDING_METRICS_DIR / "model_status.json", {})
    embedding_history_raw = _read_jsonl(EMBEDDING_METRICS_DIR / "history.jsonl", limit=50)

    # Read worker status from data/metrics/ (operational metrics)
    worker_status = _load_json(METRICS_DIR / "worker_status.json", {})

    # Read CV analysis metrics from data/metrics/
    cv_analysis_history = _read_jsonl(METRICS_DIR / "cv_analysis_history.jsonl", limit=6)
    cv_analysis_stats = _load_json(METRICS_DIR / "cv_analysis_stats.json", {})

    # Read search metrics
    search_history_raw = _read_all_jsonl(SEARCH_HISTORY_LOG)


    return {
        "model_services": _format_model_services(model_status),
        "embedding_worker": _format_embedding_worker(worker_status.get("embedding", {})),
        "embedding_stats": _calculate_embedding_stats(embedding_history_raw),
        "embedding_history": _format_embedding_history(embedding_history_raw[:6]),  # Show last 6
        "cv_stats": _format_cv_stats(cv_analysis_stats),
        "cv_history": _format_cv_history(cv_analysis_history),
        "search_stats": _format_search_stats(search_history_raw),
        "search_history": _format_search_history(search_history_raw[-10:]), # last 10
    }


def _format_search_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {
            "total_searches": "0", "keyword_searches": "0", "semantic_searches": "0",
            "avg_duration": "0ms", "avg_keyword_duration": "0ms", "avg_semantic_duration": "0ms",
            "error_rate": "0.0%"
        }

    df = pd.DataFrame(entries)
    total_searches = len(df)
    keyword_searches = len(df[df['search_type'] == 'keyword'])
    semantic_searches = len(df[df['search_type'] == 'semantic'])

    avg_duration = df['duration_ms'].mean()
    avg_keyword_duration = df[df['search_type'] == 'keyword']['duration_ms'].mean()
    avg_semantic_duration = df[df['search_type'] == 'semantic']['duration_ms'].mean()

    error_rate = len(df[df['status'] == 'error']) / total_searches if total_searches > 0 else 0

    return {
        "total_searches": _format_number(total_searches),
        "keyword_searches": _format_number(keyword_searches),
        "semantic_searches": _format_number(semantic_searches),
        "avg_duration": _format_ms(avg_duration),
        "avg_keyword_duration": _format_ms(avg_keyword_duration),
        "avg_semantic_duration": _format_ms(avg_semantic_duration),
        "error_rate": _format_percentage(error_rate)
    }

def _format_search_history(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for entry in reversed(entries): # newest first
        query = entry.get("query", {})
        query_str = ""
        if entry['search_type'] == 'keyword':
            parts = []
            if query.get('keywords'): parts.append(f"KW: '{query['keywords']}'")
            if query.get('location'): parts.append(f"Loc: '{query['location']}'")
            if query.get('skills'): parts.append(f"Skills: '{query['skills']}'")
            query_str = ", ".join(parts)
        elif entry['search_type'] == 'semantic':
            query_str = f"CV: {query.get('filename', 'N/A')}"

        rows.append({
            "timestamp": _format_timestamp(entry.get("timestamp")),
            "type": entry.get("search_type", "N/A").title(),
            "query": query_str,
            "results": _format_number(entry.get("results_count")),
            "duration": _format_ms(entry.get("duration_ms", 0)),
            "status": _status_label(entry.get("status")),
            "status_class": _status_badge_class(entry.get("status")),
        })
    return rows


def _format_model_services(model_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format per-model service status cards.

    New structure: model_status is keyed by model_name (e.g., "all-mpnet-base-v2")
    Each entry contains: service_type, status, embedding_dim, last_used
    """
    service_icons = {
        "SentenceTransformer": "deployed_code",
        "OpenAI": "cloud",
        "Cohere": "cloud_circle",
        "vLLM": "dns"
    }

    cards = []
    for model_name, info in model_status.items():
        service_type = info.get("service_type", "Unknown")
        status = (info.get("status") or "OFFLINE").upper()
        embedding_dim = info.get("embedding_dim")

        cards.append({
            "title": model_name,
            "icon": service_icons.get(service_type, "smart_toy"),
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "models": f"{service_type} (dim: {embedding_dim})" if embedding_dim else service_type,
            "last_check": _format_timestamp(info.get("last_used"))
        })

    # If no models registered, show placeholder
    if not cards:
        cards.append({
            "title": "No models loaded",
            "icon": "deployed_code",
            "status": "Offline",
            "status_class": "status-offline",
            "models": "Run embedding generation first",
            "last_check": "N/A"
        })

    return cards


def _format_embedding_worker(worker_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format embedding worker status."""
    status = (worker_info.get("status") or "IDLE").upper()
    details = worker_info.get("details", {})

    return {
        "status": _status_label(status),
        "status_class": _status_badge_class(status),
        "description": _describe_embedding_worker(status, details),
        "last_run": _format_timestamp(details.get("last_run") or worker_info.get("updated_at"))
    }


def _describe_embedding_worker(status: str, details: Dict[str, Any]) -> str:
    """Generate description for embedding worker."""
    if status == "RUNNING":
        jobs = details.get("jobs_processing")
        return f"Processing {_format_number(jobs)} jobs" if jobs else "Generating embeddings"
    if status == "IDLE":
        if details.get("jobs_processed"):
            return f"Last run processed {_format_number(details['jobs_processed'])} jobs"
        return "Waiting for jobs"
    return details.get("message", "Requires attention")


def _calculate_embedding_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics from embedding history."""
    if not entries:
        return {
            "total_embeddings": "0",
            "total_jobs": "0",
            "avg_speed": "0",
            "last_run": "N/A"
        }

    total_embeddings = sum(e.get("total_embeddings", 0) for e in entries)
    total_jobs = sum(e.get("num_jobs", 0) for e in entries)
    speeds = [e.get("embeddings_per_second", 0) for e in entries if e.get("embeddings_per_second")]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0

    # Get most recent timestamp
    latest = entries[0] if entries else {}
    last_run = _format_timestamp(latest.get("timestamp"))

    return {
        "total_embeddings": _format_number(total_embeddings),
        "total_jobs": _format_number(total_jobs),
        "avg_speed": f"{_format_float(avg_speed)} emb/s",
        "last_run": last_run
    }


def _format_embedding_history(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format embedding generation history."""
    rows = []
    for entry in entries:
        ts = entry.get("timestamp")
        job_id = f"emb_{_parse_timestamp(ts).strftime('%Y%m%d_%H%M%S')}"
        status = entry.get("status")

        rows.append({
            "job_id": job_id,
            "model": entry.get("model_name", "Unknown"),
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "jobs_processed": _format_number(entry.get("num_jobs")),
            "total_embeddings": _format_number(entry.get("total_embeddings")),
            "duration": _format_duration(entry.get("duration_seconds", 0)),
            "speed": f"{_format_float(entry.get('embeddings_per_second', 0))} emb/s",
            "completed": _format_timestamp(ts)
        })

    if not rows:
        rows.append({
            "job_id": "emb_none",
            "model": "No runs yet",
            "status": "Idle",
            "status_class": "status-idle",
            "jobs_processed": "-",
            "total_embeddings": "-",
            "duration": "-",
            "speed": "-",
            "completed": "-"
        })

    return rows


def _format_cv_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format aggregate CV analysis statistics."""
    return {
        "total_analyzed": _format_number(stats.get("total_analyzed", 0)),
        "average_duration": _format_duration(stats.get("average_duration_seconds", 0)),
        "error_rate": _format_percentage(stats.get("error_rate", 0)),
        "updated_at": _format_timestamp(stats.get("updated_at"))
    }


def _format_cv_history(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format CV analysis history."""
    rows = []
    for entry in entries:
        ts = entry.get("timestamp")
        cv_id = entry.get("cv_id", f"cv_{_parse_timestamp(ts).strftime('%Y%m%d_%H%M%S')}")
        status = entry.get("status")

        rows.append({
            "cv_id": cv_id[:12] if len(cv_id) > 12 else cv_id,  # Truncate long IDs
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "skills_found": _format_number(entry.get("skills_extracted")),
            "jobs_matched": _format_number(entry.get("jobs_matched")),
            "duration": _format_duration(entry.get("duration_seconds", 0)),
            "model": entry.get("model_used", "N/A"),
            "timestamp": _format_timestamp(ts)
        })

    if not rows:
        rows.append({
            "cv_id": "No activity yet",
            "status": "Idle",
            "status_class": "status-idle",
            "skills_found": "-",
            "jobs_matched": "-",
            "duration": "-",
            "model": "-",
            "timestamp": "-"
        })

    return rows
