"""Career Guidance Agent: personalised advice from Agents 1-3 plus live job-market data from Apify.

Apify is used only to look up current listings for a target role. The search input contains the
role (and an optional country code) and nothing from the resume. If Apify is unavailable the
guidance is still built from the other agents' outputs and is labelled as fallback guidance.
"""
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from agents.skill_gap_analysis import extract_required_skills

logger = logging.getLogger(__name__)

# Actor "misceres/indeed-scraper" (Indeed Scraper). Input fields (position, country,
# maxItemsPerSearch) and output fields (positionName, company, location, salary, jobType, url,
# postedAt, description) were checked against its published input schema and dataset schema.
ACTOR_ID = "misceres~indeed-scraper"
API_BASE = "https://api.apify.com/v2"
MAX_ITEMS = 10  # listings per run
MAX_CHARGE_USD = 0.25  # hard spend cap sent to Apify for the run
ACTOR_RUN_TIMEOUT_SECONDS = 90
HTTP_TIMEOUT_SECONDS = 120
MAX_ROLE_CHARS = 80
MAX_DESCRIPTION_CHARS = 5000
MAX_ITEMS_RETURNED = 10

Runner = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]


class ApifyError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "error": message, "data": None}


# ---------------------------------------------------------------- Apify access

def _run_actor(token: str, actor_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One synchronous Actor run returning dataset items. Isolated so tests can mock it."""
    query = urllib.parse.urlencode({
        "timeout": ACTOR_RUN_TIMEOUT_SECONDS,
        "maxItems": MAX_ITEMS,
        "limit": MAX_ITEMS,
        "maxTotalChargeUsd": MAX_CHARGE_USD,
        "clean": "true",
    })
    url = f"{API_BASE}/actors/{ACTOR_ID}/run-sync-get-dataset-items?{query}"
    request = urllib.request.Request(
        url,
        data=json.dumps(actor_input).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        if status == 429:
            raise ApifyError("rate_limited", "Apify rate limit reached. Try again later.")
        if status in (401, 403):
            raise ApifyError("auth_error", "Apify rejected the API token. Check APIFY_API_TOKEN.")
        if status == 402:
            raise ApifyError("quota_exceeded", "Apify usage limit or credit exhausted.")
        if status == 404:
            raise ApifyError("actor_unavailable", "The Apify Actor was not found or is not accessible.")
        if status == 408:
            raise ApifyError("timeout", "The Apify Actor run took too long.")
        if status == 400:
            raise ApifyError("actor_error", "The Apify Actor rejected the request or failed.")
        raise ApifyError("api_error", f"Apify request failed (HTTP {status}).")
    except (socket.timeout, TimeoutError):
        raise ApifyError("timeout", "The Apify request timed out.")
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
            raise ApifyError("timeout", "The Apify request timed out.")
        raise ApifyError("api_error", "Could not reach Apify (network error).")
    except OSError:
        raise ApifyError("api_error", "Could not reach Apify (network error).")

    try:
        items = json.loads(body)
    except ValueError:
        raise ApifyError("invalid_response", "Apify returned a response that is not valid JSON.")
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ApifyError("invalid_response", "Apify returned an unexpected response shape.")
    return items


def _is_placeholder(token: Optional[str]) -> bool:
    if not token or not token.strip():
        return True
    low = token.strip().lower()
    return low.startswith("your_") or "your_apify" in low or "your-token" in low


# ---------------------------------------------------------------- helpers

def sanitize_role(value: Any) -> Optional[str]:
    """Keep only a short job-title-like string; drop anything that looks like contact data."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"https?://\S+|\S+@\S+", " ", value)
    text = re.sub(r"\d{5,}", " ", text)
    text = re.sub(r"[^\w\s+#./&-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:MAX_ROLE_CHARS].strip()
    return text or None


def _data(result: Any) -> Optional[Dict[str, Any]]:
    if isinstance(result, dict) and result.get("status") in ("ok", "partial") and isinstance(result.get("data"), dict):
        return result["data"]
    return None


def _short(value: Any, limit: int = 120) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value).strip()
    return text[:limit] if text else None


def summarize_market(items: List[Dict[str, Any]], resume_skills: List[str]) -> Dict[str, Any]:
    """Aggregate live listings. Every figure here is derived from the supplied items only."""
    counts: Counter = Counter()
    listings: List[Dict[str, Any]] = []
    salaries: List[str] = []
    job_types: Counter = Counter()
    for item in items:
        description = item.get("description")
        if isinstance(description, str) and description:
            counts.update(set(extract_required_skills(description[:MAX_DESCRIPTION_CHARS])))
        job_type = item.get("jobType")
        for jt in (job_type if isinstance(job_type, list) else [job_type]):
            if isinstance(jt, str) and jt:
                job_types[jt] += 1
        salary = _short(item.get("salary"), 60)
        if salary:
            salaries.append(salary)
        url = item.get("url")
        if len(listings) < MAX_ITEMS_RETURNED:
            listings.append({
                "title": _short(item.get("positionName")),
                "company": _short(item.get("company")),
                "location": _short(item.get("location")),
                "salary": salary,
                "job_type": _short(item.get("jobType"), 60),
                "posted": _short(item.get("postedAt"), 40),
                "url": url if isinstance(url, str) and url.startswith(("http://", "https://")) else None,
            })
    total = len(items)
    have = {s.strip().lower() for s in resume_skills}
    top = [
        {"skill": s, "listings": c, "percent": round(100 * c / total), "in_resume": s in have}
        for s, c in counts.most_common(10)
    ] if total else []
    return {
        "source": "apify_live",
        "actor": ACTOR_ID,
        "listing_count": total,
        "top_skills": top,
        "salary_mentions": salaries[:5],
        "job_types": dict(job_types),
        "sample_listings": listings,
    }


# ---------------------------------------------------------------- guidance

def _prioritized_skills(gaps: Optional[Dict[str, Any]], market: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for rec in (gaps or {}).get("recommended_to_learn", []) or []:
        if isinstance(rec, dict) and rec.get("skill") and rec["skill"] not in seen:
            seen.add(rec["skill"])
            out.append({"skill": rec["skill"], "priority": rec.get("priority", "medium"),
                        "reason": rec.get("reason") or "Identified as a gap for the target job.",
                        "evidence": "skill_gap_analysis"})
    if market:
        for entry in market["top_skills"]:
            if not entry["in_resume"] and entry["skill"] not in seen and entry["percent"] >= 30:
                seen.add(entry["skill"])
                out.append({"skill": entry["skill"], "priority": "medium",
                            "reason": f"Mentioned in {entry['listings']} of {market['listing_count']} live listings and not on your resume.",
                            "evidence": "apify_live"})
    return out


def _project_ideas(role: Optional[str], skills: List[Dict[str, Any]]) -> List[str]:
    names = [s["skill"] for s in skills[:3]]
    ideas: List[str] = []
    for name in names[:2]:
        ideas.append(f"Build a small, finished project that uses {name} in a realistic way, and publish it with a README explaining the problem and your design choices.")
    if len(names) >= 2:
        ideas.append(f"Combine {names[0]} and {names[1]} in one portfolio project that mirrors typical {role or 'target role'} work.")
    if not ideas:
        ideas.append(f"Build one portfolio project that reflects day-to-day {role or 'target role'} work and document the outcome.")
    return ideas


def _resume_improvements(resume: Optional[Dict[str, Any]], gaps: Optional[Dict[str, Any]]) -> List[str]:
    tips: List[str] = []
    if not resume:
        return ["Provide a parsed resume to get resume-specific feedback."]
    if not resume.get("professional_summary"):
        tips.append("Add a short professional summary targeted at your desired role.")
    if not resume.get("projects"):
        tips.append("Add a projects section; it is currently empty.")
    if not resume.get("certifications"):
        tips.append("Add relevant certifications or courses if you have any (do not list ones you have not completed).")
    if not resume.get("work_experience"):
        tips.append("Add work experience, internships or volunteering with measurable outcomes.")
    if len(resume.get("skills") or []) < 5:
        tips.append("List more of your genuine skills; fewer than five were detected.")
    contact = resume.get("contact") or {}
    missing_contact = [k for k in ("email", "linkedin", "github") if not contact.get(k)]
    if missing_contact:
        tips.append("Consider adding contact/profile links that were not detected: " + ", ".join(missing_contact) + ".")
    matched = [m.get("skill") for m in (gaps or {}).get("matched", []) if isinstance(m, dict) and m.get("skill")]
    if matched:
        tips.append("Make these job-relevant skills prominent near the top: " + ", ".join(matched[:6]) + ".")
    return tips


def _job_search_suggestions(job_match: Optional[Dict[str, Any]], market: Optional[Dict[str, Any]],
                            role: Optional[str]) -> List[str]:
    tips: List[str] = []
    matches = [m for m in (job_match or {}).get("matches", []) if isinstance(m, dict)]
    best = matches[0].get("match_score") if matches else None
    if isinstance(best, (int, float)):
        if best >= 70:
            tips.append(f"Your best match scores {best}/100: apply now to your top-ranked matches.")
        elif best >= 40:
            tips.append(f"Your best match scores {best}/100: apply to the closest matches while you close the highest-priority skill gaps.")
        else:
            tips.append(f"Your best match scores {best}/100: consider adjacent or entry-level roles and build the priority skills first.")
    elif not matches:
        tips.append("No job matches were available; broaden the target role or keywords and re-run matching.")
    if market and market["listing_count"]:
        tips.append(f"Review the {market['listing_count']} live listings for {role} below to tailor your resume keywords.")
    else:
        tips.append("Search job boards for your target role and note the skills that repeat across postings.")
    tips.append("Tailor your resume headline and skills list to each application using the wording in the posting.")
    return tips


def build_guidance(resume, job_match, gaps, market, role) -> Dict[str, Any]:
    skills = _prioritized_skills(gaps, market)
    snapshot = {
        "best_match_score": None,
        "skill_coverage_percent": (gaps or {}).get("coverage_percent"),
        "missing_skills": list((gaps or {}).get("missing", []) or []),
    }
    matches = [m for m in (job_match or {}).get("matches", []) if isinstance(m, dict)]
    if matches:
        snapshot["best_match_score"] = matches[0].get("match_score")
    return {
        "guidance_source": "apify_live" if market else "fallback_no_market_data",
        "target_role": role,
        "profile_snapshot": snapshot,
        "market": market or {"source": "unavailable"},
        "skills_to_prioritize": skills,
        "project_ideas": _project_ideas(role, skills),
        "job_search_suggestions": _job_search_suggestions(job_match, market, role),
        "resume_improvements": _resume_improvements(resume, gaps),
    }


def generate_guidance(
    resume_analysis: Any = None,
    job_match: Any = None,
    skill_gaps: Any = None,
    target_role: Optional[str] = None,
    country: Optional[str] = None,
    runner: Optional[Runner] = None,
) -> Dict[str, Any]:
    """Build career guidance. Never raises; Apify failures degrade to fallback guidance."""
    resume = _data(resume_analysis)
    matches_data = _data(job_match)
    gaps = _data(skill_gaps)
    if not (resume or matches_data or gaps):
        return _error("no_input", "No usable output from the resume, job matching or skill gap agents.")

    role = sanitize_role(target_role)
    if not role and matches_data:
        for m in matches_data.get("matches", []) or []:
            if isinstance(m, dict) and sanitize_role(m.get("title")):
                role = sanitize_role(m["title"])
                break

    def fallback(code: str, message: str) -> Dict[str, Any]:
        return {"status": "partial", "error_code": code, "warning": message + " Guidance below uses only your resume, job match and skill gap results.",
                "data": build_guidance(resume, matches_data, gaps, None, role)}

    if not role:
        return fallback("no_search_term", "No target role available for a market lookup.")

    if runner is None:
        load_dotenv()
        token = os.getenv("APIFY_API_TOKEN")
        if _is_placeholder(token):
            return fallback("missing_api_token", "APIFY_API_TOKEN is not set.")
        runner_fn: Runner = _run_actor
    else:
        token, runner_fn = "mock", runner

    actor_input: Dict[str, Any] = {"position": role, "maxItemsPerSearch": MAX_ITEMS}
    if isinstance(country, str) and re.fullmatch(r"[A-Za-z]{2}", country.strip()):
        actor_input["country"] = country.strip().upper()

    try:
        items = runner_fn(token, actor_input)
        if not isinstance(items, list):
            raise ApifyError("invalid_response", "Apify returned an unexpected response shape.")
    except ApifyError as exc:
        logger.warning("Apify market lookup failed (%s).", exc.code)
        return fallback(exc.code, exc.message)
    except Exception:
        logger.warning("Apify market lookup failed (unexpected error).")
        return fallback("api_error", "Unexpected error while contacting Apify.")

    if not items:
        return fallback("no_results", "Apify returned no listings for this role.")
    skills = (resume or {}).get("skills") or []
    market = summarize_market(items[:MAX_ITEMS], [s for s in skills if isinstance(s, str)])
    return {"status": "ok", "data": build_guidance(resume, matches_data, gaps, market, role)}


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: reads resume_analysis, job_match, skill_gaps (+ optional target_role/country)."""
    result = generate_guidance(
        resume_analysis=state.get("resume_analysis"),
        job_match=state.get("job_match"),
        skill_gaps=state.get("skill_gaps"),
        target_role=state.get("target_role"),
        country=state.get("country"),
    )
    return {"career_guidance": result}