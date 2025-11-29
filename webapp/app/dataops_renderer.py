"""
Data Ops Renderer

Keeps the static HTML intact for designers while replacing dynamic content
server-side by updating DOM nodes via lxml.
"""

from pathlib import Path
from typing import Dict, List

from lxml import html as lhtml

from .dataops_loader import build_dataops_context


def render_dataops_page(template_path: Path) -> str:
    """Read template HTML and inject live data via DOM manipulation."""
    doc = lhtml.fromstring(template_path.read_text(encoding="utf-8"))
    context = build_dataops_context()

    _inject_live_status(doc, context["workers"])
    _inject_source_health(doc, context["sources"])
    _inject_collection_history(doc, context["collection_history"])
    _inject_processing_history(doc, context["processing_history"])

    return lhtml.tostring(doc, encoding="unicode")


def _inject_live_status(doc, workers: List[Dict[str, str]]) -> None:
    try:
        container = doc.get_element_by_id("live-status-cards")
    except KeyError:
        return

    template = _extract_template(container, "status-card")
    container.clear()

    for worker in workers:
        card = _clone_element(template)
        card.attrib.pop("data-template", None)
        _set_slot_text(card, "title", worker["title"])
        status_badge = _get_slot(card, "status-badge")
        if status_badge is not None:
            status_badge.attrib["class"] = f"status-badge {worker['status_class']}"
        _set_slot_text(card, "status-icon", worker["icon"])
        _set_slot_text(card, "status-text", worker["status"])
        _set_slot_text(card, "description", worker["description"])
        _set_slot_text(card, "timestamp", f"Last update: {worker['last_run']}")
        container.append(card)


def _inject_source_health(doc, sources: List[Dict[str, str]]) -> None:
    try:
        tbody = doc.get_element_by_id("source-health-body")
    except KeyError:
        return

    template = _extract_template(tbody, "source-row")
    tbody.clear()

    for source in sources:
        row = _clone_element(template)
        row.attrib.pop("data-template", None)
        _set_slot_text(row, "icon", source["icon"])
        _set_slot_text(row, "name", source["name"])
        status_badge = _get_slot(row, "status-badge")
        if status_badge is not None:
            status_badge.attrib["class"] = f"status-badge {source['status_class']}"
        _set_slot_text(row, "status-text", source["status"])
        _set_slot_text(row, "last-check", source["last_check"])
        _set_slot_text(row, "response-time", source["response_time"])
        _set_slot_text(row, "rate-limit", source["rate_limit"])
        tbody.append(row)


def _inject_collection_history(doc, rows: List[Dict[str, str]]) -> None:
    try:
        tbody = doc.get_element_by_id("collection-history-body")
    except KeyError:
        return

    template = _extract_template(tbody, "collection-row")
    tbody.clear()

    for entry in rows:
        row = _clone_element(template)
        row.attrib.pop("data-template", None)
        _set_slot_text(row, "job-id", entry["job_id"])
        _set_slot_text(row, "source", entry["source"])
        status_badge = _get_slot(row, "status-badge")
        if status_badge is not None:
            status_badge.attrib["class"] = f"status-badge {entry['status_class']}"
            status_badge.text = entry["status"]
        _set_slot_text(row, "records", entry["records"])
        _set_slot_text(row, "duration", entry["duration"])
        _set_slot_text(row, "started", entry["started"])
        button = _get_slot(row, "log-button")
        if button is not None:
            if entry.get("log_path"):
                button.attrib.pop("disabled", None)
            else:
                button.attrib["disabled"] = "disabled"
        tbody.append(row)


def _inject_processing_history(doc, rows: List[Dict[str, str]]) -> None:
    try:
        tbody = doc.get_element_by_id("processing-history-body")
    except KeyError:
        return

    template = _extract_template(tbody, "processing-row")
    tbody.clear()

    for entry in rows:
        row = _clone_element(template)
        row.attrib.pop("data-template", None)
        _set_slot_text(row, "job-id", entry["job_id"])
        status_badge = _get_slot(row, "status-badge")
        if status_badge is not None:
            status_badge.attrib["class"] = f"status-badge {entry['status_class']}"
            status_badge.text = entry["status"]
        _set_slot_text(row, "records-in", entry["records_in"])
        _set_slot_text(row, "records-out", entry["records_out"])
        _set_slot_text(row, "duration", entry["duration"])
        _set_slot_text(row, "completed", entry["completed"])

        output_cell = _get_slot(row, "output-cell")
        if output_cell is not None:
            output_cell.clear()
            if entry.get("output"):
                link = lhtml.Element(
                    "a",
                    attrib={
                        "href": entry["output"],
                        "class": "btn btn-primary",
                        "style": "padding: 0.25rem 0.75rem; font-size: 0.875rem; text-decoration: none;"
                    }
                )
                link.text = "Download"
                output_cell.append(link)
            else:
                span = lhtml.Element("span", attrib={"style": "color: var(--color-silver);"})
                span.text = "N/A"
                output_cell.append(span)

        tbody.append(row)


def _extract_template(parent, template_name: str):
    if parent is None:
        return lhtml.Element("div")
    node = parent.xpath(f'.//*[@data-template="{template_name}"]')
    if node:
        return node[0]
    return parent[0] if len(parent) else lhtml.Element("div")


def _clone_element(element):
    return lhtml.fromstring(lhtml.tostring(element))


def _get_slot(element, slot: str):
    result = element.xpath(f'.//*[@data-slot="{slot}"]')
    return result[0] if result else None


def _set_slot_text(element, slot: str, text: str) -> None:
    target = _get_slot(element, slot)
    if target is not None:
        target.text = str(text)
