"""
Metrics Utilities

Helpers for persisting collection/processing metadata so the Data Ops hub
can render real pipeline activity without shipping large datasets.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIRNAME = "metrics"


def _metrics_dir(data_dir: Optional[Path] = None) -> Path:
    """Ensure metrics directory exists and return it."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    metrics_dir = data_dir / METRICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        json.dump(record, file)
        file.write("\n")


def update_worker_status(
    worker: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path] = None
) -> None:
    """Persist latest worker status (collection / processing)."""
    metrics_dir = _metrics_dir(data_dir)
    status_path = metrics_dir / "worker_status.json"
    payload = _load_json(status_path, {})

    payload[worker] = {
        "status": status.upper(),
        "updated_at": datetime.utcnow().isoformat(),
        "details": details or {}
    }

    _write_json(status_path, payload)


def log_collection_run(
    source: str,
    profile: str,
    result: Dict[str, Any],
    data_dir: Optional[Path] = None
) -> None:
    """Append a collection run entry for history tables."""
    metrics_dir = _metrics_dir(data_dir)
    history_path = metrics_dir / "collection_history.jsonl"

    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "source": source,
        "profile": profile,
        "status": result.get("status"),
        "jobs_collected": result.get("jobs_collected", 0),
        "jobs_deduplicated": result.get("jobs_deduplicated", 0),
        "duration_seconds": result.get("duration_seconds", 0),
        "api_calls": result.get("api_calls", 0),
        "parameters": result.get("parameters", {}),
        "filepath": result.get("filepath"),
        "job_ids": result.get("job_ids", [])
    }

    _append_jsonl(history_path, entry)


def log_source_health(
    source: str,
    status: str,
    *,
    response_time_ms: Optional[float] = None,
    rate_limit_used: Optional[int] = None,
    rate_limit_total: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path] = None
) -> None:
    """Persist latest health snapshot for upstream data sources."""
    metrics_dir = _metrics_dir(data_dir)
    path = metrics_dir / "source_health.json"
    payload = _load_json(path, {})

    payload[source] = {
        "status": status.upper(),
        "last_check": datetime.utcnow().isoformat(),
        "response_time_ms": response_time_ms,
        "rate_limit_used": rate_limit_used,
        "rate_limit_total": rate_limit_total,
        "details": details or {}
    }

    _write_json(path, payload)


def log_processing_run(
    result: Dict[str, Any],
    data_dir: Optional[Path] = None
) -> None:
    """Append consolidation/processing run metadata."""
    metrics_dir = _metrics_dir(data_dir)
    history_path = metrics_dir / "processing_history.jsonl"

    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "status": result.get("status"),
        "duration_seconds": result.get("duration_seconds", 0),
        "records_in": result.get("records_in", 0),
        "records_in_by_source": result.get("records_in_by_source", {}),
        "records_out": result.get("records_out", 0),
        "output_path": result.get("output_path")
    }

    _append_jsonl(history_path, entry)


# ============================================================================
# ML/AI Metrics
# ============================================================================

def update_model_status(
    model_name: str,
    status: str,
    service_type: str = "SentenceTransformer",
    embedding_dim: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None
) -> None:
    """
    Persist per-model status to models/embeddings/metrics/.

    Args:
        model_name: Specific model name (e.g., "all-mpnet-base-v2")
        status: Model status (ONLINE, OFFLINE, ERROR)
        service_type: Type of service (SentenceTransformer, OpenAI, etc.)
        embedding_dim: Embedding dimension for this model
        details: Additional model-specific details
    """
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"

    # Create metrics directory within embeddings folder
    metrics_dir = models_dir / "embeddings" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    status_path = metrics_dir / "model_status.json"

    payload = _load_json(status_path, {})

    # Per-model status tracking
    payload[model_name] = {
        "service_type": service_type,
        "status": status.upper(),
        "embedding_dim": embedding_dim,
        "last_used": datetime.utcnow().isoformat(),
        "details": details or {}
    }

    _write_json(status_path, payload)


def log_embedding_run(
    result: Dict[str, Any],
    models_dir: Optional[Path] = None
) -> None:
    """Append embedding generation run metadata to models/embeddings/metrics/."""
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"

    # Create metrics directory within embeddings folder
    metrics_dir = models_dir / "embeddings" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "history.jsonl"

    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "status": result.get("status"),
        "model_name": result.get("model_name"),
        "embedding_dim": result.get("embedding_dim"),
        "num_jobs": result.get("num_jobs"),
        "num_embeddings_per_job": result.get("num_embeddings_per_job"),
        "total_embeddings": result.get("total_embeddings"),
        "duration_seconds": result.get("duration_seconds", 0),
        "embeddings_per_second": result.get("embeddings_per_second", 0),
        "output_path": result.get("output_path")
    }

    _append_jsonl(history_path, entry)


def update_embedding_worker_status(
    status: str,
    details: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path] = None
) -> None:
    """Update embedding worker status."""
    update_worker_status("embedding", status, details, data_dir)


def log_cv_analysis(
    result: Dict[str, Any],
    data_dir: Optional[Path] = None
) -> None:
    """Append CV analysis activity."""
    metrics_dir = _metrics_dir(data_dir)
    history_path = metrics_dir / "cv_analysis_history.jsonl"

    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "status": result.get("status"),
        "cv_id": result.get("cv_id"),
        "skills_extracted": result.get("skills_extracted", 0),
        "jobs_matched": result.get("jobs_matched", 0),
        "duration_seconds": result.get("duration_seconds", 0),
        "model_used": result.get("model_used"),
        "error": result.get("error")
    }

    _append_jsonl(history_path, entry)


def update_cv_analysis_stats(
    total_analyzed: int,
    average_duration: float,
    error_rate: float,
    data_dir: Optional[Path] = None
) -> None:
    """Update aggregate CV analysis statistics."""
    metrics_dir = _metrics_dir(data_dir)
    stats_path = metrics_dir / "cv_analysis_stats.json"

    payload = {
        "total_analyzed": total_analyzed,
        "average_duration_seconds": average_duration,
        "error_rate": error_rate,
        "updated_at": datetime.utcnow().isoformat()
    }

    _write_json(stats_path, payload)
