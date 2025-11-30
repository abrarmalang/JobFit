"""
CV Visualization Module

Creates matplotlib charts for CV analysis results.
Charts are rendered as base64-encoded images for embedding in HTML.
"""

import io
import base64
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette matching Jobfit brand
COLORS = {
    'primary': '#667EEA',      # cosmic-purple
    'secondary': '#FD79A8',    # pink accent
    'success': '#10B981',
    'warning': '#F59E0B',
    'info': '#3B82F6',
    'dark': '#1E293B',
    'light': '#F8FAFC',
    'gray': '#64748B',
}


def create_ats_score_chart(ats_score: int) -> str:
    """
    Create a circular gauge chart for ATS score.

    Args:
        ats_score: ATS score (0-100)

    Returns:
        Base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    # Define score ranges and colors
    if ats_score >= 85:
        color = COLORS['success']
        label = 'Excellent'
    elif ats_score >= 70:
        color = COLORS['info']
        label = 'Good'
    elif ats_score >= 50:
        color = COLORS['warning']
        label = 'Moderate'
    else:
        color = COLORS['secondary']
        label = 'Needs Work'

    # Create donut chart
    sizes = [ats_score, 100 - ats_score]
    colors_chart = [color, COLORS['light']]
    explode = (0.05, 0)

    wedges, texts = ax.pie(
        sizes,
        explode=explode,
        colors=colors_chart,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    # Add score text in center
    ax.text(0, 0.1, f'{ats_score}', ha='center', va='center',
            fontsize=48, fontweight='bold', color=color)
    ax.text(0, -0.25, label, ha='center', va='center',
            fontsize=16, color=COLORS['gray'])

    ax.set_title('ATS Readiness Score', fontsize=18, fontweight='bold',
                 color=COLORS['dark'], pad=20)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_skills_chart(skills: List[Dict[str, Any]], max_skills: int = 10) -> str:
    """
    Create a horizontal bar chart of top skills by category.

    Args:
        skills: List of skill dictionaries
        max_skills: Maximum number of skills to display

    Returns:
        Base64-encoded PNG image
    """
    if not skills:
        return _create_empty_chart("No skills detected")

    # Group skills by category and count
    category_counts = {}
    for skill in skills[:max_skills]:
        category = skill.get('category', 'Other')
        category_counts[category] = category_counts.get(category, 0) + 1

    # Sort by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [cat for cat, _ in sorted_categories]
    counts = [count for _, count in sorted_categories]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(categories) * 0.6)))

    # Color gradient
    colors_list = [COLORS['primary'], COLORS['info'], COLORS['success'],
                   COLORS['warning'], COLORS['secondary'], COLORS['gray']]
    bar_colors = [colors_list[i % len(colors_list)] for i in range(len(categories))]

    y_pos = np.arange(len(categories))
    ax.barh(y_pos, counts, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Number of Skills', fontsize=12, fontweight='bold')
    ax.set_title('Skills by Category', fontsize=16, fontweight='bold',
                 color=COLORS['dark'], pad=20)

    # Add value labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(count + 0.1, i, f'{count}', va='center', fontsize=10,
                fontweight='bold', color=COLORS['dark'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_proficiency_chart(skills: List[Dict[str, Any]], top_n: int = 8) -> str:
    """
    Create a radar/spider chart showing skill proficiency levels.

    Args:
        skills: List of skill dictionaries
        top_n: Number of top skills to display

    Returns:
        Base64-encoded PNG image
    """
    if not skills:
        return _create_empty_chart("No skills to analyze")

    # Take top N skills
    top_skills = skills[:min(top_n, len(skills))]

    # Map proficiency to numeric values
    proficiency_map = {'Expert': 3, 'Intermediate': 2, 'Familiar': 1}
    skill_names = [skill['name'] for skill in top_skills]
    skill_values = [proficiency_map.get(skill.get('proficiency', 'Familiar'), 1)
                    for skill in top_skills]

    # Number of variables
    num_vars = len(skill_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    skill_values += skill_values[:1]  # Complete the circle
    angles += angles[:1]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, skill_values, 'o-', linewidth=2, color=COLORS['primary'], label='Your Level')
    ax.fill(angles, skill_values, alpha=0.25, color=COLORS['primary'])

    # Fix axis to go from 0 to 3
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Familiar', 'Intermediate', 'Expert'], fontsize=10)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skill_names, fontsize=11)

    ax.set_title('Top Skills Proficiency', fontsize=16, fontweight='bold',
                 color=COLORS['dark'], pad=30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_gap_analysis_chart(gap_analysis: Dict[str, Any]) -> str:
    """
    Create a chart showing skill gap analysis for target role.

    Args:
        gap_analysis: Gap analysis results

    Returns:
        Base64-encoded PNG image
    """
    required_skills = gap_analysis.get('required_skills', [])
    matching_skills = gap_analysis.get('matching_skills', [])
    missing_skills = gap_analysis.get('missing_skills', [])

    if not required_skills:
        return _create_empty_chart("No target role specified")

    # Create data for stacked bar chart
    categories = ['Required Skills']
    matched = [len(matching_skills)]
    missing = [len(missing_skills)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked horizontal bar
    y_pos = np.arange(len(categories))
    p1 = ax.barh(y_pos, matched, color=COLORS['success'], alpha=0.8, label='Matched')
    p2 = ax.barh(y_pos, missing, left=matched, color=COLORS['warning'], alpha=0.8, label='Missing')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Number of Skills', fontsize=12, fontweight='bold')

    # Add percentage text
    match_pct = gap_analysis.get('match_percentage', 0)
    target_role = gap_analysis.get('target_role', 'Target Role')

    ax.set_title(f'Skill Match Analysis: {target_role}\n{match_pct:.1f}% Match',
                 fontsize=16, fontweight='bold', color=COLORS['dark'], pad=20)

    # Add value labels
    ax.text(matched[0] / 2, 0, f'{matched[0]} matched',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    if missing[0] > 0:
        ax.text(matched[0] + missing[0] / 2, 0, f'{missing[0]} missing',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.legend(loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_skill_comparison_chart(gap_analysis: Dict[str, Any]) -> str:
    """
    Create a detailed comparison chart of required vs. current skills.

    Args:
        gap_analysis: Gap analysis results

    Returns:
        Base64-encoded PNG image
    """
    required_skills = gap_analysis.get('required_skills', [])
    matching_skills = set(gap_analysis.get('matching_skills', []))

    if not required_skills:
        return _create_empty_chart("No required skills data")

    # Create data
    skill_names = required_skills
    has_skill = [1 if skill in matching_skills else 0 for skill in skill_names]

    fig, ax = plt.subplots(figsize=(10, max(6, len(skill_names) * 0.5)))

    # Color code bars
    colors_list = [COLORS['success'] if has else COLORS['warning'] for has in has_skill]

    y_pos = np.arange(len(skill_names))
    ax.barh(y_pos, [1] * len(skill_names), color=colors_list, alpha=0.7,
            edgecolor='white', linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(skill_names, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    # Add checkmarks and crosses (using text instead of unicode symbols for better compatibility)
    for i, (skill, has) in enumerate(zip(skill_names, has_skill)):
        symbol = 'YES' if has else 'NO'
        color = COLORS['success'] if has else COLORS['warning']
        ax.text(0.5, i, symbol, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    target_role = gap_analysis.get('target_role', 'Target Role')
    ax.set_title(f'Required Skills for {target_role}',
                 fontsize=16, fontweight='bold', color=COLORS['dark'], pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], alpha=0.7, label='Have Skill'),
        Patch(facecolor=COLORS['warning'], alpha=0.7, label='Need Skill')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_section_coverage_chart(sections: Dict[str, float]) -> str:
    """
    Create a pie chart showing CV section coverage.

    Args:
        sections: Dictionary of section names and their coverage percentages

    Returns:
        Base64-encoded PNG image
    """
    if not sections:
        return _create_empty_chart("No section data available")

    # Filter out very small values (less than 3%)
    filtered_sections = {k: v for k, v in sections.items() if v >= 3.0}

    if not filtered_sections:
        return _create_empty_chart("No sections detected")

    # Sort by value for better visualization
    sorted_items = sorted(filtered_sections.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use brand colors
    colors_list = [COLORS['primary'], COLORS['info'], COLORS['success'],
                   COLORS['warning'], COLORS['secondary']]
    pie_colors = [colors_list[i % len(colors_list)] for i in range(len(labels))]

    # Create pie chart with better label positioning
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        textprops=dict(fontweight='bold', fontsize=14),
        wedgeprops=dict(edgecolor='white', linewidth=2),
        pctdistance=0.75,
        labeldistance=1.15
    )

    # Make percentage text white, bold, and uniform size
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)

    # Make label text bold, darker, and uniform size
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
        text.set_color(COLORS['dark'])

    ax.set_title('CV Content Distribution', fontsize=18, fontweight='bold',
                 color=COLORS['dark'], pad=20)

    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64-encoded PNG string.

    Args:
        fig: Matplotlib figure

    Returns:
        Base64-encoded PNG string with data URL prefix
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_base64}'


def _create_empty_chart(message: str) -> str:
    """
    Create an empty chart with a message.

    Args:
        message: Message to display

    Returns:
        Base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center',
            fontsize=16, color=COLORS['gray'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_all_charts(cv_analysis: Dict[str, Any], gap_analysis: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Generate all visualization charts for CV analysis.

    Args:
        cv_analysis: CV analysis results
        gap_analysis: Optional gap analysis results

    Returns:
        Dictionary mapping chart names to base64-encoded images
    """
    charts = {
        'ats_score': create_ats_score_chart(cv_analysis.get('ats_score', 0)),
        'skills_by_category': create_skills_chart(cv_analysis.get('skills', [])),
        'skill_proficiency': create_proficiency_chart(cv_analysis.get('skills', [])),
    }

    # Add section coverage if available
    sections = cv_analysis.get('sections', {}).get('sections', {})
    if sections:
        charts['section_coverage'] = create_section_coverage_chart(sections)

    # Add gap analysis charts if provided
    if gap_analysis:
        charts['gap_analysis'] = create_gap_analysis_chart(gap_analysis)
        charts['skill_comparison'] = create_skill_comparison_chart(gap_analysis)

    return charts
