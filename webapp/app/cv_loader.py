"""
CV Review Loader

Handles CV analysis data processing and preparation for rendering.
Similar to dataops_loader.py and mlops_loader.py pattern.
"""

from typing import Dict, Any, Optional
from .cv_analyzer_llm import analyze_cv_text, analyze_with_target_role
from .cv_visualizer import generate_all_charts
import os


def build_cv_context(cv_text: str, target_role: Optional[str] = None) -> Dict[str, Any]:
    """
    Build context data for CV review page rendering.

    Args:
        cv_text: Extracted CV text
        target_role: Optional target job role for gap analysis

    Returns:
        Dictionary containing all data needed for CV review page
    """

    # Check if any LLM API key is available in environment
    use_llm = any([
        os.getenv('GROQ_API_KEY'),
        os.getenv('GEMINI_API_KEY'),
        os.getenv('OPENAI_API_KEY'),
        os.getenv('ANTHROPIC_API_KEY'),
        os.getenv('HF_API_KEY')
    ])

    # Perform CV analysis (will use LLM if API key available, otherwise fallback)
    cv_analysis = analyze_cv_text(cv_text, use_llm=use_llm)

    # Perform gap analysis if target role provided
    gap_analysis = None
    if target_role:
        gap_analysis = analyze_with_target_role(cv_text, target_role, cv_analysis)

    # Generate all visualization charts
    charts = generate_all_charts(cv_analysis, gap_analysis)

    # Build suggestions with proper structure
    suggestions = [
        {
            "icon": sug.get("icon", "info"),
            "color": sug.get("color", "info"),
            "title": sug.get("title", ""),
            "description": sug.get("description", ""),
        }
        for sug in cv_analysis.get("suggestions", [])
    ]

    # Categorize skills by proficiency
    skills = cv_analysis.get("skills", [])
    expert_skills = [s for s in skills if s.get("proficiency") == "Expert"]
    intermediate_skills = [s for s in skills if s.get("proficiency") == "Intermediate"]
    familiar_skills = [s for s in skills if s.get("proficiency") == "Familiar"]

    context = {
        # Overall scores
        "ats_score": cv_analysis.get("ats_score", 0),
        "analysis_summary": cv_analysis.get("analysis_summary", ""),

        # Skills data
        "skills": skills,
        "expert_skills": expert_skills,
        "intermediate_skills": intermediate_skills,
        "familiar_skills": familiar_skills,
        "total_skills": len(skills),

        # Suggestions
        "suggestions": suggestions,
        "suggestion_count": len(suggestions),

        # Charts (base64-encoded images)
        "charts": charts,

        # Gap analysis (if available)
        "has_gap_analysis": gap_analysis is not None,
        "gap_analysis": gap_analysis or {},

        # Metadata
        "cv_word_count": cv_analysis.get("sections", {}).get("total_words", 0),
        "cv_text": cv_text,
    }

    return context
