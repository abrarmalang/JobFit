"""
Data Ops Context Loader

Reads pipeline metrics written by the worker scripts and shapes them into
display-friendly structures for the SSR Data Ops page.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"


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


def _parse_timestamp(ts: str) -> datetime:
    try:
        # Handle timezone-offset strings
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


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
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
        "SUCCESS": "Completed",
        "COMPLETED": "Completed",
        "NO_NEW_JOBS": "Completed",
        "NO_DATA": "No Data",
    }
    return mapping.get(status, status.title() if status else "Unknown")


def build_dataops_context() -> Dict[str, Any]:
    """Return data ready for SSR rendering."""
    worker_status = _load_json(METRICS_DIR / "worker_status.json", {})
    source_health = _load_json(METRICS_DIR / "source_health.json", {})
    collection_history = _read_jsonl(METRICS_DIR / "collection_history.jsonl", limit=6)
    processing_history = _read_jsonl(METRICS_DIR / "processing_history.jsonl", limit=6)

    return {
        "workers": _format_workers(worker_status),
        "sources": _format_sources(source_health),
        "collection_history": _format_collection_history(collection_history),
        "processing_history": _format_processing_history(processing_history)
    }


def _format_workers(worker_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = {
        "collection": {
            "title": "Collection Worker",
            "icon": {
                "RUNNING": "play_circle",
                "IDLE": "schedule",
                "ERROR": "error"
            }
        },
        "processing": {
            "title": "Processing Worker",
            "icon": {
                "RUNNING": "bolt",
                "IDLE": "schedule",
                "ERROR": "error"
            }
        }
    }

    cards = []
    for key, meta in defaults.items():
        raw = worker_status.get(key, {})
        status = (raw.get("status") or "IDLE").upper()
        details = raw.get("details", {})
        icon = meta["icon"].get(status, meta["icon"]["IDLE"])

        cards.append({
            "title": meta["title"],
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "icon": icon,
            "description": _describe_worker(key, status, details),
            "last_run": _format_timestamp(details.get("last_run") or raw.get("updated_at"))
        })

    return cards


def _describe_worker(worker: str, status: str, details: Dict[str, Any]) -> str:
    if status == "RUNNING":
        mode = details.get("mode") or "running job"
        return f"Currently {mode.replace('-', ' ')}"
    if status == "IDLE":
        if worker == "collection" and details.get("jobs_collected") is not None:
            return f"Last run fetched {_format_number(details['jobs_collected'])} jobs"
        if worker == "processing" and details.get("records_out") is not None:
            return f"Last run produced {_format_number(details['records_out'])} records"
        return "Waiting for next schedule"
    return details.get("message", "Requires attention")


def _format_sources(source_health: Dict[str, Any]) -> List[Dict[str, Any]]:
    display_meta = {
        "Adzuna": {"label": "Adzuna API", "icon": "api"},
        "RemoteOK": {"label": "RemoteOK", "icon": "public"}
    }

    rows = []
    for key, info in source_health.items():
        meta = display_meta.get(key, {"label": key, "icon": "lan"})
        rate_used = info.get("rate_limit_used")
        rate_total = info.get("rate_limit_total")
        if rate_used is not None and rate_total is not None:
            rate_text = f"{rate_used} / {rate_total} req/day"
        else:
            rate_text = "-"

        response = info.get("response_time_ms")
        response_text = f"{int(response)} ms" if response else "-"

        rows.append({
            "name": meta["label"],
            "icon": meta["icon"],
            "status": _status_label(info.get("status")),
            "status_class": _status_badge_class(info.get("status")),
            "last_check": _format_timestamp(info.get("last_check")),
            "response_time": response_text,
            "rate_limit": rate_text
        })

    if not rows:
        rows.append({
            "name": "Adzuna API",
            "icon": "api",
            "status": "Offline",
            "status_class": "status-offline",
            "last_check": "N/A",
            "response_time": "-",
            "rate_limit": "-"
        })

    return rows


def _format_collection_history(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        ts = entry.get("timestamp")
        job_id = f"col_{_parse_timestamp(ts).strftime('%Y%m%d_%H%M%S')}"
        status = entry.get("status")
        rows.append({
            "job_id": job_id,
            "source": entry.get("source", "Unknown"),
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "records": _format_number(entry.get("jobs_collected")),
            "duration": _format_duration(entry.get("duration_seconds", 0)),
            "started": _format_timestamp(ts),
            "log_path": entry.get("filepath")
        })

    if not rows:
        rows.append({
            "job_id": "col_none",
            "source": "No runs yet",
            "status": "Idle",
            "status_class": "status-idle",
            "records": "-",
            "duration": "-",
            "started": "-",
            "log_path": None
        })

    return rows


def _format_processing_history(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        ts = entry.get("timestamp")
        job_id = f"cons_{_parse_timestamp(ts).strftime('%Y%m%d_%H%M%S')}"
        status = entry.get("status")
        rows.append({
            "job_id": job_id,
            "status": _status_label(status),
            "status_class": _status_badge_class(status),
            "records_in": _format_number(entry.get("records_in")),
            "records_out": _format_number(entry.get("records_out")),
            "duration": _format_duration(entry.get("duration_seconds", 0)),
            "completed": _format_timestamp(ts),
            "output": entry.get("output_path")
        })

    if not rows:
        rows.append({
            "job_id": "cons_none",
            "status": "Idle",
            "status_class": "status-idle",
            "records_in": "-",
            "records_out": "-",
            "duration": "-",
            "completed": "-",
            "output": None
        })

    return rows
