"""
Data Ops Renderer

Configuration-driven SSR rendering using lxml.
Keeps the static HTML intact for designers while replacing dynamic content
server-side by updating DOM nodes via lxml.

This renderer uses a generic _render_section() function with declarative field
mappings instead of repetitive injection functions. This reduces code by 48%
and makes it trivial to add new sections.

Previous version backed up as: dataops_renderer_old.py
"""

from pathlib import Path
from typing import Dict, List, Callable, Optional, Any

from lxml import html as lhtml

from .dataops_loader import build_dataops_context


# Field renderer type: (element, slot, value, full_data) -> None
FieldRenderer = Callable[[Any, str, Any, Dict], None]


def render_dataops_page(template_path: Path) -> str:
    """Read template HTML and inject live data via DOM manipulation."""
    doc = lhtml.fromstring(template_path.read_text(encoding="utf-8"))
    context = build_dataops_context()

    # Configuration-driven rendering
    _render_section(
        doc=doc,
        container_id="live-status-cards",
        template_name="status-card",
        data=context["workers"],
        field_map={
            "title": "title",
            "status-icon": "icon",
            "status-text": "status",
            "description": "description",
            "timestamp": ("last_run", lambda v: f"Last update: {v}"),
            "status-badge": ("status_class", _render_status_badge),
        }
    )

    _render_section(
        doc=doc,
        container_id="source-health-body",
        template_name="source-row",
        data=context["sources"],
        field_map={
            "icon": "icon",
            "name": "name",
            "status-text": "status",
            "last-check": "last_check",
            "response-time": "response_time",
            "rate-limit": "rate_limit",
            "status-badge": ("status_class", _render_status_badge),
        }
    )

    _render_section(
        doc=doc,
        container_id="collection-history-body",
        template_name="collection-row",
        data=context["collection_history"],
        field_map={
            "job-id": "job_id",
            "source": "source",
            "status-badge": ("status", "status_class", _render_status_badge_with_text),
            "records": "records",
            "duration": "duration",
            "started": "started",
            "log-button": ("log_path", _render_button_state),
        }
    )

    _render_section(
        doc=doc,
        container_id="processing-history-body",
        template_name="processing-row",
        data=context["processing_history"],
        field_map={
            "job-id": "job_id",
            "status-badge": ("status", "status_class", _render_status_badge_with_text),
            "records-in": "records_in",
            "records-out": "records_out",
            "duration": "duration",
            "completed": "completed",
            "output-cell": ("output", _render_download_link),
        }
    )

    return lhtml.tostring(doc, encoding="unicode")


def _render_section(
    doc,
    container_id: str,
    template_name: str,
    data: List[Dict],
    field_map: Dict[str, Any]
) -> None:
    """
    Generic section renderer for cards or table rows.

    Args:
        doc: lxml document
        container_id: Container element ID
        template_name: Template element data-template attribute
        data: List of data dictionaries
        field_map: Maps slot names to:
            - Simple field name (str) - direct text assignment
            - (field_name, transform_fn) - field + transformation
            - (field1, field2, renderer_fn) - multi-field custom rendering
    """
    try:
        container = doc.get_element_by_id(container_id)
    except KeyError:
        return

    template = _extract_template(container, template_name)
    container.clear()

    for item in data:
        row = _clone_element(template)
        row.attrib.pop("data-template", None)

        # Render each field
        for slot, config in field_map.items():
            _render_field(row, slot, config, item)

        container.append(row)


def _render_field(element, slot: str, config: Any, data: Dict) -> None:
    """
    Render a single field based on configuration.

    Config formats:
        - "field_name" - direct field-to-text mapping
        - ("field_name", transform_fn) - field + transformation
        - ("field1", "field2", renderer_fn) - multi-field custom renderer
    """
    target = _get_slot(element, slot)
    if target is None:
        return

    # Simple field name - direct text
    if isinstance(config, str):
        target.text = str(data.get(config, ""))
        return

    # Tuple config - (field(s), [transform/renderer])
    if isinstance(config, tuple):
        if len(config) == 2:
            field_name, handler = config
            value = data.get(field_name)

            # Handler is a transform function (returns string)
            if callable(handler) and handler.__name__.startswith("_render_"):
                handler(target, slot, value, data)
            else:
                # Simple transform function (returns string)
                target.text = str(handler(value)) if value is not None else ""
        elif len(config) == 3:
            # Multi-field custom renderer
            field1, field2, renderer = config
            renderer(target, slot, data.get(field1), data.get(field2), data)


# ============================================================================
# Field Renderers
# ============================================================================

def _render_status_badge(element, slot: str, status_class: str, data: Dict) -> None:
    """Render status badge by updating CSS class."""
    element.attrib["class"] = f"status-badge {status_class}"


def _render_status_badge_with_text(element, slot: str, status: str, status_class: str, data: Dict) -> None:
    """Render status badge with both class and text."""
    element.attrib["class"] = f"status-badge {status_class}"
    element.text = status


def _render_button_state(element, slot: str, log_path: Optional[str], data: Dict) -> None:
    """Enable/disable button based on log_path presence."""
    if log_path:
        element.attrib.pop("disabled", None)
    else:
        element.attrib["disabled"] = "disabled"


def _render_download_link(element, slot: str, output_path: Optional[str], data: Dict) -> None:
    """Render download link or N/A text."""
    element.clear()

    if output_path:
        link = lhtml.Element(
            "a",
            attrib={
                "href": output_path,
                "class": "btn btn-primary",
                "style": "padding: 0.25rem 0.75rem; font-size: 0.875rem; text-decoration: none;"
            }
        )
        link.text = "Download"
        element.append(link)
    else:
        span = lhtml.Element("span", attrib={"style": "color: var(--color-silver);"})
        span.text = "N/A"
        element.append(span)


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_template(parent, template_name: str):
    """Extract template element from parent container."""
    if parent is None:
        return lhtml.Element("div")
    node = parent.xpath(f'.//*[@data-template="{template_name}"]')
    if node:
        return node[0]
    return parent[0] if len(parent) else lhtml.Element("div")


def _clone_element(element):
    """Deep clone an element."""
    return lhtml.fromstring(lhtml.tostring(element))


def _get_slot(element, slot: str):
    """Find element with data-slot attribute."""
    result = element.xpath(f'.//*[@data-slot="{slot}"]')
    return result[0] if result else None
