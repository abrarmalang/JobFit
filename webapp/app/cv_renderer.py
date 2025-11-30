"""
CV Review Renderer

SSR rendering using lxml for CV review page.
Follows the same pattern as dataops_renderer.py and mlops_renderer.py.
"""

from pathlib import Path
from typing import Dict, Any
from lxml import html as lhtml


def render_cv_review_page(template_path: Path, context: Dict[str, Any]) -> str:
    """
    Render CV review page with analysis results.

    Args:
        template_path: Path to HTML template
        context: Context data from cv_loader.build_cv_context()

    Returns:
        Rendered HTML string
    """
    doc = lhtml.fromstring(template_path.read_text(encoding="utf-8"))

    # Hide upload section, show results section
    upload_section = doc.get_element_by_id("uploadSection", None)
    if upload_section is not None:
        upload_section.set("style", "display: none;")

    results_section = doc.get_element_by_id("resultsSection", None)
    if results_section is not None:
        results_section.set("style", "display: block; padding-top: 3rem;")

    # Render ATS score
    _render_ats_score(doc, context)

    # Render skills
    _render_skills(doc, context)

    # Render suggestions
    _render_suggestions(doc, context)

    # Render charts
    _render_charts(doc, context)

    # Render gap analysis if available
    if context.get("has_gap_analysis"):
        _render_gap_analysis(doc, context)

    return lhtml.tostring(doc, encoding="unicode")


def _render_ats_score(doc, context: Dict[str, Any]):
    """Render ATS score section."""
    score = context.get("ats_score", 0)
    summary = context.get("analysis_summary", "")

    # Update score number
    score_elements = doc.xpath('//div[@style and contains(@style, "font-size: 3.5rem")]')
    if score_elements:
        score_elements[0].text = str(score)

    # Update summary text
    summary_elements = doc.xpath('//p[strong[contains(text(), "ATS compatibility")]]')
    if summary_elements:
        # Parse and update with new summary
        summary_elements[0].clear()
        # Use lxml's HTML parsing to preserve <strong> tags
        summary_frag = lhtml.fragment_fromstring(f'<p>{summary}</p>')
        for child in summary_frag:
            summary_elements[0].append(child)
        # Copy text content
        if summary_frag.text:
            summary_elements[0].text = summary_frag.text

    # Update circular progress indicator
    # Calculate stroke-dashoffset: circumference * (1 - score/100)
    # circumference = 2 * pi * r = 2 * 3.14159 * 80 = 502.65
    circumference = 502.65
    offset = circumference * (1 - score / 100)

    circle_elements = doc.xpath('//circle[@stroke-dashoffset]')
    if circle_elements:
        circle_elements[0].set("stroke-dashoffset", str(offset))


def _render_skills(doc, context: Dict[str, Any]):
    """Render skills section."""
    skills = context.get("skills", [])

    # Find skills container - ALWAYS find it to clear placeholder content
    skills_container = doc.xpath('//div[@style and contains(@style, "display: flex; flex-wrap: wrap")]')
    if not skills_container:
        return

    container = skills_container[0]
    container.clear()  # Remove placeholder skills

    # If no skills found, show message
    if not skills:
        no_skills_p = lhtml.Element("p")
        no_skills_p.set("style", "color: var(--color-graphite); font-style: italic;")
        no_skills_p.text = "No technical skills detected. This may be a non-technical CV."
        container.append(no_skills_p)
        return

    # Add skills as badges
    for i, skill in enumerate(skills[:15]):  # Limit to top 15 skills
        skill_name = skill.get("name", "")
        proficiency = skill.get("proficiency", "Familiar")

        # Use gradient for expert skills, light background for others
        if proficiency == "Expert":
            style = "background: var(--gradient-cosmic); color: white; padding: 0.6rem 1.2rem; border-radius: 2rem; font-weight: 600; font-size: 0.9375rem;"
        else:
            style = "background: rgba(102,126,234,0.15); color: var(--color-cosmic-purple); padding: 0.6rem 1.2rem; border-radius: 2rem; font-weight: 600; font-size: 0.9375rem;"

        skill_span = lhtml.Element("span")
        skill_span.set("style", style)
        skill_span.text = skill_name
        container.append(skill_span)


def _render_suggestions(doc, context: Dict[str, Any]):
    """Render improvement suggestions section."""
    suggestions = context.get("suggestions", [])

    if not suggestions:
        return

    # Find suggestions container
    suggestions_container = doc.xpath('//div[@style and contains(@style, "display: grid; gap: 1.5rem")]')
    if not suggestions_container:
        return

    container = suggestions_container[0]
    container.clear()  # Remove placeholder suggestions

    # Add each suggestion
    for suggestion in suggestions:
        icon = suggestion.get("icon", "info")
        color = suggestion.get("color", "info")
        title = suggestion.get("title", "")
        description = suggestion.get("description", "")

        # Map color names to CSS variables
        color_map = {
            "warning": "var(--color-warning)",
            "info": "var(--color-info)",
            "success": "var(--color-success)",
            "cosmic-purple": "var(--color-cosmic-purple)",
        }
        color_value = color_map.get(color, "var(--color-info)")

        # Create suggestion card
        card_div = lhtml.Element("div")
        card_div.set("style", f"display: flex; gap: 1rem; padding: 1.5rem; background: var(--color-ghost); border-radius: 0.75rem; border-left: 4px solid {color_value};")

        # Icon
        icon_div = lhtml.Element("div")
        icon_div.set("class", "material-icons")
        icon_div.set("style", f"color: {color_value}; font-size: 1.5rem;")
        icon_div.text = icon
        card_div.append(icon_div)

        # Content
        content_div = lhtml.Element("div")

        # Title
        title_h4 = lhtml.Element("h4")
        title_h4.set("style", "margin-bottom: 0.5rem; color: var(--color-obsidian); font-size: 1.125rem;")
        title_h4.text = title
        content_div.append(title_h4)

        # Description
        desc_p = lhtml.Element("p")
        desc_p.set("style", "margin: 0; color: var(--color-graphite); line-height: 1.6;")
        desc_p.text = description
        content_div.append(desc_p)

        card_div.append(content_div)
        container.append(card_div)


def _render_charts(doc, context: Dict[str, Any]):
    """Render visualization charts."""
    charts = context.get("charts", {})

    if not charts:
        return

    # Find results section to add charts
    results_section = doc.get_element_by_id("resultsSection", None)
    if results_section is None:
        return

    # Find container div inside results section
    container_divs = results_section.xpath('.//div[@class="container"]')
    if not container_divs:
        return

    main_container = container_divs[0]

    # Create charts section after the existing content
    charts_section = lhtml.Element("div")
    charts_section.set("style", "margin-top: 3rem;")

    # Section header
    header_div = lhtml.Element("div")
    header_div.set("style", "margin-bottom: 2rem; text-align: center;")

    header_h2 = lhtml.Element("h2")
    header_h2.set("class", "section-title")
    header_h2.set("style", "margin-bottom: 0.5rem;")
    header_h2.text = "Visual Analysis"
    header_div.append(header_h2)

    header_p = lhtml.Element("p")
    header_p.set("class", "section-description")
    header_p.text = "Comprehensive insights from your CV"
    header_div.append(header_p)

    charts_section.append(header_div)

    # Chart grid
    chart_grid = lhtml.Element("div")
    chart_grid.set("style", "display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; max-width: 1200px; margin: 0 auto;")

    # Add each chart
    chart_titles = {
        "ats_score": "ATS Readiness Score",
        "skills_by_category": "Skills by Category",
        "skill_proficiency": "Skill Proficiency Levels",
        "section_coverage": "CV Content Distribution",
        "gap_analysis": "Skill Gap Analysis",
        "skill_comparison": "Required Skills Comparison",
    }

    for chart_key, chart_base64 in charts.items():
        if not chart_base64:
            continue

        # Chart card
        card = lhtml.Element("div")
        card.set("class", "feature-card")
        card.set("style", "padding: 2rem; text-align: center;")

        # Chart title
        chart_title = lhtml.Element("h3")
        chart_title.set("style", "margin-bottom: 1.5rem; font-size: 1.25rem; color: var(--color-obsidian);")
        chart_title.text = chart_titles.get(chart_key, "Analysis Chart")
        card.append(chart_title)

        # Chart image
        img = lhtml.Element("img")
        img.set("src", chart_base64)
        img.set("alt", chart_titles.get(chart_key, "Chart"))
        img.set("style", "max-width: 100%; height: auto; border-radius: 0.5rem;")
        card.append(img)

        chart_grid.append(card)

    charts_section.append(chart_grid)

    # Insert charts section before the action buttons
    # Find the action buttons div
    action_buttons = main_container.xpath('.//div[@style and contains(@style, "margin-top: 3rem; text-align: center")]')
    if action_buttons:
        # Insert before action buttons
        parent = action_buttons[0].getparent()
        parent.insert(parent.index(action_buttons[0]), charts_section)
    else:
        # Append to end
        main_container.append(charts_section)


def _render_gap_analysis(doc, context: Dict[str, Any]):
    """Render gap analysis section if target role was specified."""
    gap_analysis = context.get("gap_analysis", {})

    if not gap_analysis:
        return

    # Find results section
    results_section = doc.get_element_by_id("resultsSection", None)
    if results_section is None:
        return

    container_divs = results_section.xpath('.//div[@class="container"]')
    if not container_divs:
        return

    main_container = container_divs[0]

    # Create gap analysis section
    gap_section = lhtml.Element("div")
    gap_section.set("style", "max-width: 900px; margin: 3rem auto 0;")

    # Card
    card = lhtml.Element("div")
    card.set("class", "feature-card")
    card.set("style", "padding: 2.5rem;")

    # Header
    header_div = lhtml.Element("div")
    header_div.set("style", "display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;")

    icon = lhtml.Element("div")
    icon.set("class", "material-icons")
    icon.set("style", "font-size: 2rem; color: var(--color-cosmic-purple);")
    icon.text = "compare_arrows"
    header_div.append(icon)

    title = lhtml.Element("h3")
    title.set("class", "feature-title")
    title.set("style", "margin: 0;")
    title.text = f"Gap Analysis: {gap_analysis.get('target_role', 'Target Role')}"
    header_div.append(title)

    card.append(header_div)

    # Match percentage
    match_pct = gap_analysis.get("match_percentage", 0)
    match_p = lhtml.Element("p")
    match_p.set("style", "color: var(--color-graphite); margin-bottom: 1.5rem; font-size: 1.125rem;")

    match_strong = lhtml.Element("strong")
    match_strong.text = f"{match_pct:.1f}% match"
    match_p.append(match_strong)
    match_p.tail = " with target role requirements"
    # Set text before the strong element
    match_p.text = "You have a "

    card.append(match_p)

    # Missing skills
    missing_skills = gap_analysis.get("missing_skills", [])
    if missing_skills:
        missing_label = lhtml.Element("p")
        missing_label.set("style", "font-weight: 600; margin-bottom: 1rem; color: var(--color-obsidian);")
        missing_label.text = "Skills to develop:"
        card.append(missing_label)

        skills_div = lhtml.Element("div")
        skills_div.set("style", "display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.5rem;")

        for skill in missing_skills:
            skill_badge = lhtml.Element("span")
            skill_badge.set("style", "background: var(--color-warning); color: white; padding: 0.5rem 1rem; border-radius: 1.5rem; font-weight: 600; font-size: 0.875rem;")
            skill_badge.text = skill
            skills_div.append(skill_badge)

        card.append(skills_div)

    # Recommendations
    recommendations = gap_analysis.get("recommendations", [])
    if recommendations:
        rec_label = lhtml.Element("p")
        rec_label.set("style", "font-weight: 600; margin-bottom: 1rem; color: var(--color-obsidian);")
        rec_label.text = "Recommendations:"
        card.append(rec_label)

        rec_ul = lhtml.Element("ul")
        rec_ul.set("style", "margin: 0; padding-left: 1.5rem; color: var(--color-graphite); line-height: 1.8;")

        for rec in recommendations:
            rec_li = lhtml.Element("li")
            rec_li.text = rec
            rec_ul.append(rec_li)

        card.append(rec_ul)

    gap_section.append(card)

    # Insert gap analysis after suggestions, before charts
    # Find suggestions section
    suggestions_cards = main_container.xpath('.//div[@class="feature-card" and .//h3[contains(text(), "Suggestions")]]')
    if suggestions_cards:
        parent = suggestions_cards[0].getparent().getparent()
        # Insert after the parent of suggestions card
        parent_parent = parent.getparent()
        insert_index = parent_parent.index(parent) + 1
        parent_parent.insert(insert_index, gap_section)
    else:
        # Insert before visual analysis section
        visual_sections = main_container.xpath('.//h2[contains(text(), "Visual Analysis")]')
        if visual_sections:
            parent = visual_sections[0].getparent().getparent()
            parent_parent = parent.getparent()
            insert_index = parent_parent.index(parent)
            parent_parent.insert(insert_index, gap_section)
