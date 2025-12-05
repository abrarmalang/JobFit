"""
ML Ops Renderer

Server-side rendering for ML operations monitoring using lxml.
"""

from pathlib import Path
from typing import Dict, List, Any
from lxml import etree, html

from .mlops_loader import build_mlops_context


def render_mlops_page(template_path: Path) -> str:
    """Server-side render ML metrics into mlops.html using lxml."""
    html_content = template_path.read_text(encoding="utf-8")
    doc = html.fromstring(html_content)

    context = build_mlops_context()

    # Render each section
    _render_model_services(doc, context["model_services"])
    _render_embedding_stats(doc, context.get("embedding_stats", {}))
    _render_embedding_history(doc, context.get("embedding_history", []))
    _render_training_metrics(doc, context.get("training_metrics", {}))
    _render_search_stats(doc, context.get("search_stats", {}))
    _render_search_history(doc, context.get("search_history", []))

    return html.tostring(doc, encoding="unicode", method="html", doctype="<!DOCTYPE html>")


def _render_search_stats(doc, stats: Dict) -> None:
    """Render search statistics summary cards."""
    try:
        container = doc.get_element_by_id("search-stats-summary")
        container.clear()

        cards = [
            {"label": "Total Searches", "value": stats.get("total_searches", "0")},
            {"label": "Keyword Searches", "value": stats.get("keyword_searches", "0")},
            {"label": "Semantic Searches", "value": stats.get("semantic_searches", "0")},
            {"label": "Avg Keyword Time", "value": stats.get("avg_keyword_duration", "0ms")},
            {"label": "Avg Semantic Time", "value": stats.get("avg_semantic_duration", "0ms")},
            {"label": "Error Rate", "value": stats.get("error_rate", "0.0%")},
        ]

        for card in cards:
            metric_box = etree.Element("div", attrib={"class": "metric-box"})
            value_div = etree.SubElement(metric_box, "div", attrib={"class": "metric-value"})
            value_div.text = card["value"]
            label_div = etree.SubElement(metric_box, "div", attrib={"class": "metric-label"})
            label_div.text = card["label"]
            container.append(metric_box)
    except KeyError:
        pass # Element not found

def _render_search_history(doc, history: List[Dict]) -> None:
    """Render search history table."""
    try:
        tbody = doc.get_element_by_id("search-history-tbody")
        tbody.clear()

        if not history:
            tr = etree.Element("tr")
            td = etree.SubElement(tr, "td", attrib={"colspan": "6", "style": "text-align: center;"})
            td.text = "No search activity yet."
            tbody.append(tr)
            return

        for entry in history:
            tr = etree.Element("tr")
            
            etree.SubElement(tr, "td").text = entry.get("timestamp", "N/A")
            etree.SubElement(tr, "td").text = entry.get("type", "N/A")

            td_query = etree.SubElement(tr, "td")
            td_query.text = entry.get("query", "")
            td_query.set("style", "max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;")


            etree.SubElement(tr, "td").text = entry.get("results", "-")
            etree.SubElement(tr, "td").text = entry.get("duration", "-")

            td_status = etree.SubElement(tr, "td")
            badge = etree.SubElement(td_status, "span", attrib={"class": f"status-badge {entry.get('status_class', '')}"})
            badge.text = entry.get("status", "N/A")

            tbody.append(tr)
    except KeyError:
        pass


def _render_model_services(doc, services: List[Dict]) -> None:
    """Render LLM & Service Status table."""
    try:
        tbody = doc.get_element_by_id("model-services-tbody")
        tbody.clear()

        for service in services:
            tr = etree.Element("tr")

            # Service name column with icon
            td1 = etree.SubElement(tr, "td")
            icon = etree.SubElement(td1, "span")
            icon.set("class", "material-icons")
            icon.set("style", "font-size: 1.25rem; vertical-align: middle; margin-right: 0.5rem;")
            icon.text = service["icon"]
            text_span = etree.SubElement(td1, "span")
            text_span.set("style", "vertical-align: middle;")
            text_span.text = service["title"]

            # Status badge column
            td2 = etree.SubElement(tr, "td")
            badge = etree.SubElement(td2, "span")
            badge.set("class", f"status-badge {service['status_class']}")
            badge_icon = etree.SubElement(badge, "span")
            badge_icon.set("class", "material-icons")
            badge_icon.set("style", "font-size: 0.875rem;")
            badge_icon.text = "check_circle" if "online" in service['status_class'] else "cancel"
            badge_icon.tail = f" {service['status']}"

            # Models column
            td3 = etree.SubElement(tr, "td")
            td3.text = service["models"]

            # Response time column (placeholder)
            td4 = etree.SubElement(tr, "td")
            td4.text = "-"

            # Last check column
            td5 = etree.SubElement(tr, "td")
            td5.text = service["last_check"]

            tbody.append(tr)
    except KeyError:
        pass


def _render_embedding_stats(doc, stats: Dict) -> None:
    """Render embedding statistics summary cards."""
    try:
        container = doc.get_element_by_id("embedding-stats-summary")
        container.clear()

        # Define stat cards
        cards = [
            {"label": "Total Embeddings Generated", "value": stats["total_embeddings"], "icon": "auto_awesome"},
            {"label": "Total Jobs Processed", "value": stats["total_jobs"], "icon": "work"},
            {"label": "Average Speed", "value": stats["avg_speed"], "icon": "speed"},
            {"label": "Last Run", "value": stats["last_run"], "icon": "schedule"}
        ]

        for card in cards:
            # Create stat card
            card_div = etree.Element("div")
            card_div.set("style", "background: var(--color-cloud); padding: 1.25rem; border-radius: 0.5rem;")

            # Icon and label
            header = etree.SubElement(card_div, "div")
            header.set("style", "display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;")
            icon = etree.SubElement(header, "span")
            icon.set("class", "material-icons")
            icon.set("style", "font-size: 1.25rem; color: var(--color-cosmic-purple);")
            icon.text = card["icon"]
            label = etree.SubElement(header, "span")
            label.set("style", "font-size: 0.875rem; color: var(--color-graphite);")
            label.text = card["label"]

            # Value
            value = etree.SubElement(card_div, "div")
            value.set("style", "font-size: 1.75rem; font-weight: 700; color: var(--color-charcoal);")
            value.text = card["value"]

            container.append(card_div)
    except KeyError:
        pass


def _render_embedding_history(doc, history: List[Dict]) -> None:
    """Render embedding generation history table."""
    try:
        tbody = doc.get_element_by_id("embedding-history-tbody")
        tbody.clear()

        for entry in history:
            tr = etree.Element("tr")

            # Job ID column
            td1 = etree.SubElement(tr, "td")
            code = etree.SubElement(td1, "code")
            code.set("style", "font-size: 0.875rem; color: var(--color-cosmic-purple);")
            code.text = entry["job_id"]

            # Model column
            td2 = etree.SubElement(tr, "td")
            td2.text = entry["model"]

            # Status badge column
            td3 = etree.SubElement(tr, "td")
            badge = etree.SubElement(td3, "span")
            badge.set("class", f"status-badge {entry['status_class']}")
            badge.text = entry["status"]

            # Jobs processed column
            td4 = etree.SubElement(tr, "td")
            td4.text = entry["jobs_processed"]

            # Duration column
            td5 = etree.SubElement(tr, "td")
            td5.text = entry["duration"]

            # Completed column
            td6 = etree.SubElement(tr, "td")
            td6.text = entry["completed"]

            tbody.append(tr)
    except KeyError:
        pass


def _render_training_metrics(doc, metrics: Dict) -> None:
    """Render cluster training metrics summary cards."""
    try:
        container = doc.get_element_by_id("training-metrics-summary")
        container.clear()

        cards = [
            {"label": "Jobs Clustered", "value": metrics.get("n_samples", "0")},
            {"label": "Number of Clusters", "value": metrics.get("n_clusters", "0")},
            {"label": "Silhouette Score", "value": metrics.get("silhouette_score", "N/A")},
            {"label": "Avg Cluster Size", "value": metrics.get("avg_cluster_size", "N/A")},
            {"label": "Largest Cluster", "value": metrics.get("largest_cluster", "N/A")},
            {"label": "Last Trained", "value": metrics.get("trained_at", "N/A")},
        ]

        for card in cards:
            metric_box = etree.Element("div", attrib={"class": "metric-box"})
            value_div = etree.SubElement(metric_box, "div", attrib={"class": "metric-value"})
            value_div.text = card["value"]
            label_div = etree.SubElement(metric_box, "div", attrib={"class": "metric-label"})
            label_div.text = card["label"]
            container.append(metric_box)
    except KeyError:
        pass
