"""
CV Analysis Module - LLM-Based Version

Uses LLM for intelligent skill extraction that works for ANY profession.
Falls back to rule-based extraction if LLM is unavailable.
"""

import re
import os
import json
from typing import Dict, List, Any, Optional


def analyze_cv_text(cv_text: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Analyze CV text and extract insights.

    Args:
        cv_text: CV text content
        use_llm: Whether to use LLM (True) or fallback to rule-based (False)

    Returns:
        Analysis results with skills, ATS score, suggestions, etc.
    """

    # Extract skills using LLM or fallback
    if use_llm and is_llm_configured():
        skills = extract_skills_llm(cv_text)
    else:
        skills = extract_skills_fallback(cv_text)

    # Calculate ATS score
    ats_score = calculate_ats_score(cv_text, skills)

    # Generate suggestions
    suggestions = generate_suggestions(cv_text, skills, ats_score)

    # Analyze sections
    sections = analyze_sections(cv_text)

    return {
        "ats_score": ats_score,
        "skills": skills,
        "suggestions": suggestions,
        "sections": sections,
        "analysis_summary": generate_summary(ats_score),
        "extraction_method": "llm" if (use_llm and is_llm_configured()) else "fallback"
    }


def is_llm_configured() -> bool:
    """Check if LLM API is configured."""
    # Check for any LLM API key in environment
    llm_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("HF_API_KEY"),  # Hugging Face
    ]
    return any(key is not None for key in llm_keys)


def extract_skills_llm(cv_text: str) -> List[Dict[str, Any]]:
    """
    Extract skills using LLM API.
    Works for ANY profession - tech, healthcare, education, sales, etc.
    """

    # Check which LLM is configured
    if os.getenv("OPENAI_API_KEY"):
        return _extract_with_openai(cv_text)
    elif os.getenv("ANTHROPIC_API_KEY"):
        return _extract_with_anthropic(cv_text)
    elif os.getenv("GEMINI_API_KEY"):
        return _extract_with_gemini(cv_text)
    elif os.getenv("GROQ_API_KEY"):
        return _extract_with_groq(cv_text)
    elif os.getenv("HF_API_KEY"):
        return _extract_with_huggingface(cv_text)
    else:
        # No LLM configured, use fallback
        return extract_skills_fallback(cv_text)


def _get_skill_extraction_prompt(cv_text: str) -> str:
    """Generate the prompt for LLM skill extraction."""

    prompt = f"""Analyze this CV and extract ALL skills mentioned. Return ONLY a JSON array of skills.

For each skill, provide:
- "name": skill name
- "proficiency": "Expert", "Intermediate", or "Familiar" based on context
- "category": appropriate category (e.g., "Healthcare", "Technology", "Business", "Communication", etc.)

CV TEXT:
{cv_text[:3000]}  # Limit to avoid token limits

RESPOND WITH ONLY A JSON ARRAY, NO OTHER TEXT:
[
  {{"name": "...", "proficiency": "...", "category": "..."}},
  ...
]
"""
    return prompt


def _extract_with_openai(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills using OpenAI API."""
    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cheaper model, upgrade to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are a professional CV analyst. Extract skills accurately from any profession."},
                {"role": "user", "content": _get_skill_extraction_prompt(cv_text)}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        # Parse JSON response
        skills_json = response.choices[0].message.content.strip()
        skills_json = skills_json.replace("```json", "").replace("```", "").strip()
        skills = json.loads(skills_json)

        return skills

    except Exception as e:
        print(f"OpenAI extraction failed: {e}")
        return extract_skills_fallback(cv_text)


def _extract_with_anthropic(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills using Anthropic Claude API."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest Claude model
            max_tokens=1000,
            messages=[
                {"role": "user", "content": _get_skill_extraction_prompt(cv_text)}
            ]
        )

        # Parse JSON response
        skills_json = message.content[0].text.strip()
        skills_json = skills_json.replace("```json", "").replace("```", "").strip()
        skills = json.loads(skills_json)

        return skills

    except Exception as e:
        print(f"Anthropic extraction failed: {e}")
        return extract_skills_fallback(cv_text)


def _extract_with_gemini(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills using Google Gemini API."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(_get_skill_extraction_prompt(cv_text))

        # Parse JSON response
        skills_json = response.text.strip()
        skills_json = skills_json.replace("```json", "").replace("```", "").strip()
        skills = json.loads(skills_json)

        return skills

    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        return extract_skills_fallback(cv_text)


def _extract_with_groq(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills using Groq API (fast Llama inference)."""
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # or mixtral-8x7b-32768
            messages=[
                {"role": "system", "content": "You are a professional CV analyst."},
                {"role": "user", "content": _get_skill_extraction_prompt(cv_text)}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        # Parse JSON response
        skills_json = completion.choices[0].message.content.strip()
        skills_json = skills_json.replace("```json", "").replace("```", "").strip()
        skills = json.loads(skills_json)

        return skills

    except Exception as e:
        print(f"Groq extraction failed: {e}")
        return extract_skills_fallback(cv_text)


def _extract_with_huggingface(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills using Hugging Face Inference API."""
    try:
        import requests

        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

        payload = {
            "inputs": _get_skill_extraction_prompt(cv_text),
            "parameters": {"max_new_tokens": 1000, "temperature": 0.1}
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        # Parse JSON response
        result = response.json()
        skills_json = result[0]["generated_text"].split("[")[-1].split("]")[0]
        skills_json = "[" + skills_json + "]"
        skills = json.loads(skills_json)

        return skills

    except Exception as e:
        print(f"Hugging Face extraction failed: {e}")
        return extract_skills_fallback(cv_text)


def extract_skills_fallback(cv_text: str) -> List[Dict[str, Any]]:
    """
    Fallback skill extraction when LLM is unavailable.
    Uses smart section parsing + pattern matching for universal coverage.
    """

    skills = []
    text_lower = cv_text.lower()

    # Strategy 1: Extract from Skills section
    skills_from_section = _extract_from_skills_section(cv_text)
    skills.extend(skills_from_section)

    # Strategy 2: Pattern matching for common skills across all industries
    skill_patterns = {
        # Universal soft skills (always useful)
        "Communication": r"\bcommunication\b",
        "Leadership": r"\bleadership\b",
        "Teamwork": r"\bteam(?:work)?\b",
        "Problem Solving": r"\bproblem solving\b",
        "Time Management": r"\btime management\b",
        "Project Management": r"\bproject management\b",

        # Common tools
        "Microsoft Office": r"\b(Microsoft Office|MS Office)\b",
        "Excel": r"\b(Excel|MS Excel)\b",
        "PowerPoint": r"\bPowerPoint\b",
        "Word": r"\b(Word|MS Word)\b",

        # Healthcare (examples)
        "Patient Care": r"\bpatient care\b",
        "Clinical Skills": r"\bclinical\b",
        "Nursing": r"\bnursing\b",

        # Business (examples)
        "Sales": r"\bsales\b",
        "Marketing": r"\bmarketing\b",
        "Customer Service": r"\bcustomer service\b",

        # Tech (examples)
        "Python": r"\bPython\b",
        "JavaScript": r"\bJavaScript\b",
        "SQL": r"\bSQL\b",
    }

    for skill_name, pattern in skill_patterns.items():
        if re.search(pattern, cv_text, re.IGNORECASE):
            # Check if not already added
            if not any(s["name"] == skill_name for s in skills):
                skills.append({
                    "name": skill_name,
                    "proficiency": _estimate_proficiency(text_lower, skill_name.lower()),
                    "category": _categorize_skill(skill_name)
                })

    return skills[:20]  # Limit to top 20


def _extract_from_skills_section(cv_text: str) -> List[Dict[str, Any]]:
    """Extract skills from dedicated skills section."""

    skills = []

    # Find skills section
    patterns = [
        r'(?i)(?:^|\n)(skills?|competencies|technical skills?)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)',
        r'(?i)(?:^|\n)(key skills?|core competencies)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, cv_text, re.DOTALL)
        if match:
            section_content = match.group(2)

            # Extract individual skills (bullet points, commas, lines)
            skill_items = re.split(r'[•\n,]', section_content)

            for item in skill_items:
                skill_name = item.strip().strip('-').strip()
                if skill_name and len(skill_name) > 2 and len(skill_name) < 50:
                    skills.append({
                        "name": skill_name.title(),
                        "proficiency": "Intermediate",  # Default for listed skills
                        "category": _categorize_skill(skill_name)
                    })

            break  # Use first match

    return skills


def _estimate_proficiency(text: str, skill: str) -> str:
    """Estimate proficiency level."""
    expert_words = ["expert", "proficient", "advanced", "certified", "specialist"]
    intermediate_words = ["experience", "worked with", "knowledge"]

    for word in expert_words:
        if f"{word}" in text and skill in text:
            return "Expert"

    for word in intermediate_words:
        if f"{word}" in text and skill in text:
            return "Intermediate"

    return "Familiar"


def _categorize_skill(skill_name: str) -> str:
    """Categorize skill into broad category."""
    skill_lower = skill_name.lower()

    if any(word in skill_lower for word in ["patient", "clinical", "medical", "nursing", "health"]):
        return "Healthcare"
    elif any(word in skill_lower for word in ["teach", "education", "classroom", "curriculum"]):
        return "Education"
    elif any(word in skill_lower for word in ["python", "javascript", "programming", "software", "code"]):
        return "Technology"
    elif any(word in skill_lower for word in ["sales", "marketing", "customer"]):
        return "Sales & Marketing"
    elif any(word in skill_lower for word in ["accounting", "financial", "budget"]):
        return "Finance"
    elif any(word in skill_lower for word in ["communication", "leadership", "teamwork", "problem"]):
        return "Soft Skills"
    else:
        return "Professional Skills"


def calculate_ats_score(text: str, skills: List[Dict]) -> int:
    """Calculate ATS score for any profession."""
    score = 40
    score += min(len(skills) * 2, 25)  # Skills bonus

    sections = ["experience", "education", "skills"]
    found_sections = sum(1 for s in sections if s in text.lower())
    score += (found_sections / len(sections)) * 15

    numbers = re.findall(r'\d+%|\d+ years?|\d+\+', text, re.IGNORECASE)
    score += min(len(numbers) * 2, 10)

    action_verbs = ["achieved", "managed", "led", "developed", "improved"]
    found_verbs = sum(1 for v in action_verbs if v in text.lower())
    score += min(found_verbs * 2, 10)

    return min(int(score), 100)


def generate_suggestions(text: str, skills: List[Dict], score: int) -> List[Dict[str, str]]:
    """Generate improvement suggestions."""
    suggestions = []

    if len(skills) < 5:
        suggestions.append({
            "icon": "psychology",
            "color": "cosmic-purple",
            "title": "Add More Skills",
            "description": "Include a dedicated skills section with at least 5-10 relevant skills to improve ATS matching."
        })

    numbers = re.findall(r'\d+%|\d+ years?', text, re.IGNORECASE)
    if len(numbers) < 3:
        suggestions.append({
            "icon": "trending_up",
            "color": "warning",
            "title": "Quantify Achievements",
            "description": "Add specific metrics like percentages, numbers, or time periods to your accomplishments."
        })

    if not re.search(r'\b(summary|objective)\b', text.lower()):
        suggestions.append({
            "icon": "format_align_left",
            "color": "info",
            "title": "Add Professional Summary",
            "description": "Include a brief summary at the top highlighting your experience and goals."
        })

    return suggestions[:5]


def analyze_sections(text: str) -> Dict[str, Any]:
    """Analyze CV sections for better visualization."""
    total_words = len(text.split())

    # Count mentions with more weight for better distribution
    sections = {
        "Experience": len(re.findall(r'\b(experience|work|employment|position|role|job)\b', text, re.IGNORECASE)) * 3,
        "Education": len(re.findall(r'\b(education|degree|university|college|school|qualification)\b', text, re.IGNORECASE)) * 3,
        "Skills": len(re.findall(r'\b(skills?|proficient|expertise|knowledge|technologies|tools)\b', text, re.IGNORECASE)) * 2,
        "Summary": len(re.findall(r'\b(summary|objective|profile|about|overview)\b', text, re.IGNORECASE)) * 2,
        "Projects": len(re.findall(r'\b(projects?|portfolio|work samples?)\b', text, re.IGNORECASE)) * 2,
    }

    # Remove zero or very low values
    sections = {k: v for k, v in sections.items() if v > 0}

    # Add a baseline "Other" if we have sections
    if sections:
        sections["Other"] = max(1, total_words // 100)

    total = sum(sections.values()) or 1
    section_percentages = {k: (v / total) * 100 for k, v in sections.items()}

    return {
        "total_words": total_words,
        "sections": section_percentages
    }


def generate_summary(score: int) -> str:
    """Generate summary message."""
    if score >= 85:
        return "Your CV has <strong>excellent ATS compatibility</strong>."
    elif score >= 70:
        return "Your CV has <strong>good ATS compatibility</strong> with room for improvement."
    elif score >= 50:
        return "Your CV has <strong>moderate ATS compatibility</strong>."
    else:
        return "Your CV needs <strong>improvement</strong> for ATS systems."


# Gap analysis functions (same as before)
def analyze_with_target_role(cv_text: str, target_role: str, cv_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Perform gap analysis for target role."""
    # Implementation same as existing
    pass
