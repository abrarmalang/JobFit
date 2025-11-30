"""
CV Analysis Module

Analyzes CVs using AI to extract skills, calculate ATS scores,
and provide improvement suggestions.
"""

import re
from typing import Dict, List, Any
from pathlib import Path


def analyze_cv_text(cv_text: str) -> Dict[str, Any]:
    """
    Analyze CV text and extract insights.

    For now, this uses rule-based extraction. In production,
    this would use an LLM API (OpenAI, Anthropic, etc.)
    """

    # Extract skills using keyword matching
    skills = extract_skills(cv_text)

    # Calculate ATS score based on various factors
    ats_score = calculate_ats_score(cv_text, skills)

    # Generate improvement suggestions
    suggestions = generate_suggestions(cv_text, skills, ats_score)

    # Extract section analysis for visualizations
    sections = analyze_sections(cv_text)

    return {
        "ats_score": ats_score,
        "skills": skills,
        "suggestions": suggestions,
        "sections": sections,
        "analysis_summary": generate_summary(ats_score)
    }


def extract_skills(text: str) -> List[Dict[str, Any]]:
    """Extract skills from CV text using keyword matching with context awareness."""

    # First, try to identify if this is a tech CV or not
    # Check for presence of tech-related roles/sections
    tech_indicators = [
        r'\b(software|developer|engineer|programmer|data scientist|devops|frontend|backend|full.?stack)\b',
        r'\b(skills?|technical skills?|technologies|programming)\b.*:',
    ]

    is_tech_cv = any(re.search(pattern, text, re.IGNORECASE) for pattern in tech_indicators)

    # If not a tech CV, return empty - avoid false positives
    if not is_tech_cv:
        return []

    # Common tech skills to look for (in production, use LLM)
    skill_patterns = {
        # Programming Languages
        "Python": r"\bPython\b",
        "JavaScript": r"\b(JavaScript|JS)\b",
        "TypeScript": r"\bTypeScript\b",
        "Java": r"\bJava\b",
        "C++": r"\bC\+\+\b",
        "Go": r"\bGo(?:lang)?\b",
        "Rust": r"\bRust\b",
        "Ruby": r"\bRuby\b",
        "PHP": r"\bPHP\b",
        "C#": r"\bC#\b",

        # Frameworks & Libraries
        "React": r"\bReact(?:\.js)?\b",
        "Vue": r"\bVue(?:\.js)?\b",
        "Angular": r"\bAngular\b",
        "Django": r"\bDjango\b",
        "Flask": r"\bFlask\b",
        "FastAPI": r"\bFastAPI\b",
        "Node.js": r"\bNode(?:\.js)?\b",
        "Express": r"\bExpress(?:\.js)?\b",
        "Spring": r"\bSpring\b",
        "TensorFlow": r"\bTensorFlow\b",
        "PyTorch": r"\bPyTorch\b",
        "Pandas": r"\bPandas\b",
        "NumPy": r"\bNumPy\b",
        "Scikit-learn": r"\bScikit-learn\b",

        # Databases
        "PostgreSQL": r"\b(PostgreSQL|Postgres)\b",
        "MySQL": r"\bMySQL\b",
        "MongoDB": r"\bMongoDB\b",
        "Redis": r"\bRedis\b",
        "SQL": r"\bSQL\b",
        "NoSQL": r"\bNoSQL\b",

        # Cloud & DevOps
        "AWS": r"\b(AWS|Amazon Web Services)\b",
        "Azure": r"\b(Azure|Microsoft Azure)\b",
        "GCP": r"\b(GCP|Google Cloud)\b",
        "Docker": r"\bDocker\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "CI/CD": r"\bCI/CD\b",
        "Jenkins": r"\bJenkins\b",
        "GitLab": r"\bGitLab\b",
        "GitHub Actions": r"\bGitHub Actions\b",

        # Tools & Practices
        "Git": r"\bGit\b",
        "REST API": r"\b(REST|RESTful|REST API)\b",
        "GraphQL": r"\bGraphQL\b",
        "Microservices": r"\bMicroservices\b",
        "Agile": r"\bAgile\b",
        "Scrum": r"\bScrum\b",
        "Terraform": r"\bTerraform\b",

        # ML/AI
        "Machine Learning": r"\bMachine Learning\b",
        "Deep Learning": r"\bDeep Learning\b",
        "NLP": r"\b(NLP|Natural Language Processing)\b",
        "Computer Vision": r"\bComputer Vision\b",
        "Data Science": r"\bData Science\b",
        "AI": r"\b(AI|Artificial Intelligence)\b",
    }

    found_skills = []
    text_lower = text.lower()

    for skill_name, pattern in skill_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            # Estimate proficiency based on context
            proficiency = estimate_proficiency(text_lower, skill_name.lower())
            found_skills.append({
                "name": skill_name,
                "proficiency": proficiency,
                "category": categorize_skill(skill_name)
            })

    # Sort by proficiency
    found_skills.sort(key=lambda x: x["proficiency"], reverse=True)

    return found_skills


def estimate_proficiency(text: str, skill: str) -> str:
    """Estimate skill proficiency based on context clues."""
    skill_lower = skill.lower()

    # Look for expertise indicators
    expert_indicators = [
        f"expert in {skill_lower}",
        f"{skill_lower} expert",
        f"advanced {skill_lower}",
        f"proficient in {skill_lower}",
        f"{skill_lower} architect",
        f"lead {skill_lower}",
    ]

    intermediate_indicators = [
        f"experience with {skill_lower}",
        f"worked with {skill_lower}",
        f"using {skill_lower}",
        f"{skill_lower} developer",
    ]

    for indicator in expert_indicators:
        if indicator in text:
            return "Expert"

    for indicator in intermediate_indicators:
        if indicator in text:
            return "Intermediate"

    return "Familiar"


def categorize_skill(skill_name: str) -> str:
    """Categorize skill into broad categories."""
    categories = {
        "Programming Languages": ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "Ruby", "PHP", "C#"],
        "Web Frameworks": ["React", "Vue", "Angular", "Django", "Flask", "FastAPI", "Node.js", "Express", "Spring"],
        "ML/AI": ["TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science", "AI"],
        "Databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQL", "NoSQL"],
        "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins", "GitLab", "GitHub Actions", "Terraform"],
        "Tools & Practices": ["Git", "REST API", "GraphQL", "Microservices", "Agile", "Scrum"],
    }

    for category, skills in categories.items():
        if skill_name in skills:
            return category

    return "Other"


def calculate_ats_score(text: str, skills: List[Dict]) -> int:
    """Calculate ATS readiness score (0-100)."""
    score = 50  # Base score

    # Bonus for skills (up to +20)
    skill_bonus = min(len(skills) * 1.5, 20)
    score += skill_bonus

    # Check for common CV sections (up to +15)
    sections = ["experience", "education", "skills", "summary", "projects"]
    found_sections = sum(1 for s in sections if s in text.lower())
    section_bonus = (found_sections / len(sections)) * 15
    score += section_bonus

    # Check for quantifiable achievements (up to +10)
    numbers = re.findall(r'\d+%|\d+x|\$\d+', text)
    if len(numbers) >= 3:
        score += 10
    elif len(numbers) >= 1:
        score += 5

    # Check for action verbs (up to +5)
    action_verbs = ["developed", "implemented", "led", "managed", "designed", "created", "built", "improved", "optimized"]
    found_verbs = sum(1 for v in action_verbs if v in text.lower())
    if found_verbs >= 5:
        score += 5

    return min(int(score), 100)


def generate_suggestions(text: str, skills: List[Dict], score: int) -> List[Dict[str, str]]:
    """Generate improvement suggestions based on CV analysis."""
    suggestions = []

    # Check for quantifiable achievements
    numbers = re.findall(r'\d+%|\d+x|\$\d+', text)
    if len(numbers) < 3:
        suggestions.append({
            "category": "Quantify Achievements",
            "priority": "high",
            "icon": "trending_up",
            "color": "warning",
            "title": "Quantify Your Achievements",
            "description": 'Add specific metrics and numbers to your accomplishments. For example, instead of "Improved system performance," use "Improved system performance by 40%, reducing API response time from 500ms to 300ms."'
        })

    # Check for professional summary
    if not re.search(r'\b(summary|objective|profile)\b', text.lower()):
        suggestions.append({
            "category": "Professional Summary",
            "priority": "high",
            "icon": "format_align_left",
            "color": "info",
            "title": "Add a Professional Summary",
            "description": "Include a 3-4 sentence summary at the top highlighting your experience level, key skills, and what you're looking for. This helps ATS systems and recruiters quickly understand your profile."
        })

    # Check for action verbs
    action_verbs = ["developed", "implemented", "led", "managed", "designed", "created", "built", "improved", "optimized"]
    found_verbs = sum(1 for v in action_verbs if v in text.lower())
    if found_verbs < 5:
        suggestions.append({
            "category": "Action Verbs",
            "priority": "medium",
            "icon": "check_circle",
            "color": "success",
            "title": "Use Strong Action Verbs",
            "description": 'Start bullet points with strong action verbs like "Developed," "Implemented," "Led," or "Optimized" rather than passive phrases. This makes your contributions more impactful.'
        })

    # Check for certifications section
    if not re.search(r'\b(certifications?|certificates?)\b', text.lower()):
        suggestions.append({
            "category": "Certifications",
            "priority": "medium",
            "icon": "extension",
            "color": "cosmic-purple",
            "title": "Include Relevant Certifications",
            "description": "If you have certifications (AWS, Azure, Google Cloud, etc.), create a dedicated section. These keywords boost ATS scores and demonstrate ongoing professional development."
        })

    # Check for projects section
    if not re.search(r'\b(projects?|portfolio)\b', text.lower()):
        suggestions.append({
            "category": "Projects",
            "priority": "medium",
            "icon": "code",
            "color": "info",
            "title": "Showcase Your Projects",
            "description": "Add a projects section with 2-3 key projects that demonstrate your skills. Include technologies used and measurable outcomes to strengthen your profile."
        })

    # Check for education
    if not re.search(r'\b(education|degree|university|bachelor|master)\b', text.lower()):
        suggestions.append({
            "category": "Education",
            "priority": "low",
            "icon": "school",
            "color": "success",
            "title": "Add Education Details",
            "description": "Include your educational background with degree, institution, and graduation year. This is often a required field in ATS systems."
        })

    return suggestions[:5]  # Return top 5 suggestions


def analyze_sections(text: str) -> Dict[str, Any]:
    """Analyze CV sections for visualization."""

    # Estimate word counts for different sections (simplified)
    total_words = len(text.split())

    # Simple heuristic: look for section keywords
    sections = {
        "Experience": len(re.findall(r'\b(experience|work|employment|position)\b', text, re.IGNORECASE)),
        "Education": len(re.findall(r'\b(education|degree|university|college)\b', text, re.IGNORECASE)),
        "Skills": len(re.findall(r'\b(skills?|technologies|tools)\b', text, re.IGNORECASE)),
        "Projects": len(re.findall(r'\b(projects?|portfolio)\b', text, re.IGNORECASE)),
        "Summary": len(re.findall(r'\b(summary|objective|profile|about)\b', text, re.IGNORECASE)),
    }

    # Normalize to percentages
    total = sum(sections.values()) or 1
    section_percentages = {k: (v / total) * 100 for k, v in sections.items()}

    return {
        "total_words": total_words,
        "sections": section_percentages
    }


def generate_summary(score: int) -> str:
    """Generate a summary message based on ATS score."""
    if score >= 85:
        return "Your CV has <strong>excellent ATS compatibility</strong> with strong keywords and well-structured content."
    elif score >= 70:
        return "Your CV has <strong>good ATS compatibility</strong> with room for improvement in quantifying achievements and formatting consistency."
    elif score >= 50:
        return "Your CV has <strong>moderate ATS compatibility</strong>. Focus on adding quantifiable achievements and relevant keywords."
    else:
        return "Your CV needs <strong>significant improvement</strong> for ATS systems. Consider restructuring with clear sections and adding measurable achievements."


def analyze_with_target_role(cv_text: str, target_role: str, cv_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform gap analysis between CV and target role.

    Args:
        cv_text: The CV text
        target_role: Target job role/title
        cv_analysis: Existing CV analysis

    Returns:
        Gap analysis results with missing skills and recommendations
    """

    # Define common skills for various roles
    role_skill_map = {
        "software engineer": ["Python", "JavaScript", "Git", "REST API", "SQL", "Agile"],
        "data scientist": ["Python", "Machine Learning", "Pandas", "SQL", "TensorFlow", "Data Science"],
        "frontend developer": ["JavaScript", "React", "TypeScript", "HTML", "CSS"],
        "backend developer": ["Python", "Java", "SQL", "REST API", "Docker", "PostgreSQL"],
        "devops engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Git"],
        "ml engineer": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Docker", "AWS"],
        "full stack developer": ["JavaScript", "Python", "React", "Node.js", "SQL", "REST API"],
    }

    # Normalize target role
    target_role_lower = target_role.lower()

    # Find matching role or use generic
    required_skills = []
    for role, skills in role_skill_map.items():
        if role in target_role_lower:
            required_skills = skills
            break

    # If no match, use generic tech skills
    if not required_skills:
        required_skills = ["Python", "JavaScript", "SQL", "Git", "Docker", "AWS"]

    # Get CV skills
    cv_skills = {skill["name"] for skill in cv_analysis["skills"]}

    # Calculate gaps
    missing_skills = [skill for skill in required_skills if skill not in cv_skills]
    matching_skills = [skill for skill in required_skills if skill in cv_skills]

    match_percentage = (len(matching_skills) / len(required_skills) * 100) if required_skills else 0

    return {
        "target_role": target_role,
        "required_skills": required_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "match_percentage": round(match_percentage, 1),
        "skill_gap_count": len(missing_skills),
        "recommendations": generate_gap_recommendations(missing_skills, target_role)
    }


def generate_gap_recommendations(missing_skills: List[str], target_role: str) -> List[str]:
    """Generate recommendations to close skill gaps."""
    recommendations = []

    if missing_skills:
        recommendations.append(
            f"Consider adding experience with {', '.join(missing_skills[:3])} to better match {target_role} requirements."
        )

        if len(missing_skills) > 3:
            recommendations.append(
                f"Additional skills to consider: {', '.join(missing_skills[3:6])}."
            )

        recommendations.append(
            "Highlight any relevant projects or coursework that demonstrate these skills, even if not from professional experience."
        )
    else:
        recommendations.append(
            f"Great match! Your skills align well with {target_role} requirements."
        )

    return recommendations
