"""Job Matching Agent: finds jobs via JSearch (RapidAPI) and scores them against the resume."""
import datetime
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

API_HOST = "jsearch.p.rapidapi.com"
API_URL = f"https://{API_HOST}/search"
TIMEOUT_SECONDS = 20
MAX_RESULTS = 10
MAX_QUERY_CHARS = 120

# Weights are renormalized over the components that can actually be evaluated.
WEIGHTS = {"skills": 0.5, "experience": 0.3, "qualifications": 0.2}

# Education levels: higher number = higher level.
EDUCATION_LEVELS = (
    ("high_school", 1, ("high school", "secondary school", "diploma")),
    ("associates_degree", 2, ("associate",)),
    ("bachelors_degree", 3, ("bachelor", "b.sc", "bsc", "b.tech", "btech", "b.e", "b.a", "undergraduate")),
    ("postgraduate_degree", 4, ("master", "m.sc", "msc", "m.tech", "mtech", "mba", "phd", "ph.d", "doctor", "postgraduate")),
)
JOB_EDU_FIELDS = {
    "high_school": 1,
    "associates_degree": 2,
    "bachelors_degree": 3,
    "postgraduate_degree": 4,
}

Fetcher = Callable[[Dict[str, str], str], Dict[str, Any]]


class JobApiError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "error": message, "data": None}


# ---------------------------------------------------------------- API access

def _fetch_jsearch(params: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """Single JSearch request. Isolated so tests can mock it."""
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": API_HOST},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise JobApiError("rate_limited", "JSearch rate limit or monthly quota reached. Try again later.")
        if exc.code in (401, 403):
            raise JobApiError("auth_error", "JSearch rejected the API key. Check JSEARCH_API_KEY and your RapidAPI subscription.")
        raise JobApiError("api_error", f"JSearch request failed (HTTP {exc.code}).")
    except (urllib.error.URLError, TimeoutError, OSError):
        raise JobApiError("api_error", "Could not reach the JSearch API.")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise JobApiError("invalid_response", "JSearch returned an invalid (non-JSON) response.")
    if not isinstance(parsed, dict):
        raise JobApiError("invalid_response", "JSearch returned an unexpected response.")
    return parsed


# ------------------------------------------------------------ query building

def build_query(target_role: Optional[str], job_description: Optional[str]) -> str:
    """Prefer the explicit role; otherwise use the first line of the job description."""
    for source in (target_role, job_description):
        if isinstance(source, str) and source.strip():
            first_line = source.strip().splitlines()[0].strip()
            return first_line[:MAX_QUERY_CHARS]
    return ""


# ------------------------------------------------------------------- scoring

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _resume_skills(resume: Dict[str, Any]) -> List[str]:
    skills = resume.get("skills")
    if not isinstance(skills, list):
        return []
    seen, out = set(), []
    for s in skills:
        n = _norm(str(s))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def score_skills(resume_skills: List[str], job: Dict[str, Any]) -> Dict[str, Any]:
    """Skills: required skills matched / required skills, or description-based fallback."""
    required = job.get("job_required_skills")
    description = _norm(str(job.get("job_description") or ""))
    if isinstance(required, list) and required:
        req = [_norm(str(r)) for r in required if str(r).strip()]
        matched = [r for r in req if any(r == s or _contains_term(r, s) or _contains_term(s, r) for s in resume_skills)]
        missing = [r for r in req if r not in matched]
        score = round(100 * len(matched) / len(req)) if req else None
        return {"score": score, "matched": matched, "missing": missing, "basis": "job_required_skills"}
    if not resume_skills:
        return {"score": 0 if description else None, "matched": [], "missing": [], "basis": "no_resume_skills"}
    if not description:
        return {"score": None, "matched": [], "missing": [], "basis": "no_job_text"}
    matched = [s for s in resume_skills if _contains_term(description, s)]
    # Cap the denominator so long skill lists are not unfairly penalized.
    score = round(100 * len(matched) / min(len(resume_skills), 10))
    return {"score": min(score, 100), "matched": matched, "missing": [], "basis": "skills_found_in_description"}


def _years_in(text: Any) -> Optional[int]:
    if not isinstance(text, str):
        return None
    if re.search(r"present|current|now|ongoing", text, re.I):
        return datetime.date.today().year
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


def resume_experience_months(resume: Dict[str, Any]) -> Optional[int]:
    """Total months from dated work entries (overlaps merged). None if nothing is dated."""
    spans: List[Tuple[int, int]] = []
    for item in resume.get("work_experience") or []:
        if not isinstance(item, dict):
            continue
        start, end = _years_in(item.get("start_date")), _years_in(item.get("end_date"))
        if start is not None and end is not None and end >= start:
            spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    # Year granularity only, so a single-year span counts as 12 months.
    return sum((e - s) * 12 or 12 for s, e in merged)


def score_experience(resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    info = job.get("job_required_experience")
    info = info if isinstance(info, dict) else {}
    required = info.get("required_experience_in_months")
    have = resume_experience_months(resume)
    base = {"resume_months": have, "required_months": required}
    if info.get("no_experience_required") or not isinstance(required, (int, float)) or required <= 0:
        return {**base, "score": 100 if info.get("no_experience_required") else None}
    if have is None:
        return {**base, "score": None, "note": "Resume work dates could not be determined."}
    return {**base, "score": min(100, round(100 * have / required))}


def resume_education_level(resume: Dict[str, Any]) -> int:
    best = 0
    for item in resume.get("education") or []:
        text = _norm(" ".join(str(v) for v in item.values() if v) if isinstance(item, dict) else str(item))
        for _, level, keywords in EDUCATION_LEVELS:
            if any(k in text for k in keywords):
                best = max(best, level)
    return best


def score_qualifications(resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    info = job.get("job_required_education")
    info = info if isinstance(info, dict) else {}
    # JSearch flags can mark several levels; the lowest acceptable one is the requirement.
    flagged = [lvl for key, lvl in JOB_EDU_FIELDS.items() if info.get(key)]
    required = min(flagged) if flagged else 0
    have = resume_education_level(resume)
    base = {"resume_level": have, "required_level": required}
    if not required:
        return {**base, "score": None}
    if have == 0:
        return {**base, "score": None, "note": "No recognizable education level in resume."}
    return {**base, "score": 100 if have >= required else round(100 * have / required)}


def combine(components: Dict[str, Optional[int]]) -> Optional[int]:
    used = {k: v for k, v in components.items() if v is not None}
    if not used:
        return None
    total = sum(WEIGHTS[k] for k in used)
    return round(sum(WEIGHTS[k] * v for k, v in used.items()) / total)


def score_job(resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    skills = score_skills(_resume_skills(resume), job)
    experience = score_experience(resume, job)
    quals = score_qualifications(resume, job)
    overall = combine({
        "skills": skills["score"],
        "experience": experience["score"],
        "qualifications": quals["score"],
    })
    return {
        "match_score": overall if overall is not None else 0,
        "score_breakdown": {"skills": skills, "experience": experience, "qualifications": quals},
        "weights": WEIGHTS,
    }


def _summarize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    location = ", ".join(str(x) for x in (job.get("job_city"), job.get("job_state"), job.get("job_country")) if x)
    return {
        "job_id": job.get("job_id"),
        "title": job.get("job_title"),
        "company": job.get("employer_name"),
        "location": location or None,
        "apply_link": job.get("job_apply_link"),
        "employment_type": job.get("job_employment_type"),
    }


# --------------------------------------------------------------- entry points

def match_jobs(
    resume: Optional[Dict[str, Any]],
    target_role: Optional[str] = None,
    job_description: Optional[str] = None,
    fetcher: Optional[Fetcher] = None,
) -> Dict[str, Any]:
    """Find and score jobs. Never raises; returns {"status", "data", ...}."""
    if not isinstance(resume, dict) or not resume:
        return _error("invalid_resume", "Parsed resume data is missing; run resume analysis first.")

    query = build_query(target_role, job_description)
    if not query:
        return _error("empty_input", "Provide a target role or job description.")

    load_dotenv()
    api_key = os.getenv("JSEARCH_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        return _error("missing_api_key", "JSEARCH_API_KEY is not set. Add it to your .env file.")

    params = {"query": query, "page": "1", "num_pages": "1"}
    try:
        payload = (fetcher or _fetch_jsearch)(params, api_key)
    except JobApiError as exc:
        logger.warning("JSearch call failed (%s).", exc.code)
        return _error(exc.code, exc.message)
    except Exception as exc:  # unexpected; message omitted to avoid leaking request details
        logger.warning("JSearch call failed (%s).", type(exc).__name__)
        return _error("api_error", "The JSearch request failed. Please try again.")

    jobs = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        return _error("invalid_response", "JSearch returned an unexpected response format.")

    matches = []
    for job in jobs[:MAX_RESULTS]:
        if not isinstance(job, dict):
            continue
        matches.append({**_summarize_job(job), **score_job(resume, job)})
    matches.sort(key=lambda m: m["match_score"], reverse=True)

    result: Dict[str, Any] = {"query": query, "matches": matches}
    if not matches:
        result["message"] = "No job listings found for this query."
    return {"status": "ok", "data": result}


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: reads resume_analysis (+ target_role / job_description), writes job_match."""
    analysis = state.get("resume_analysis")
    resume = analysis.get("data") if isinstance(analysis, dict) and analysis.get("status") == "ok" else None
    return {
        "job_match": match_jobs(
            resume,
            target_role=state.get("target_role"),
            job_description=state.get("job_description"),
        )
    }