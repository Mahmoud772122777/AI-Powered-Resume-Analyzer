"""Skill Gap Analysis Agent.

Deterministic comparison (exact + alias + subset matching) is done locally. Only skills that
remain unmatched are sent to the Hugging Face Inference API (feature-extraction embeddings)
for semantic comparison. Only short skill names are sent, never resume text or personal data.
"""
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Endpoint format taken from huggingface_hub's hf-inference provider:
#   POST https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
#   body {"inputs": [str, ...]} -> list of embedding vectors (one per input).
# BAAI/bge-small-en-v1.5 is the feature-extraction model listed for hf-inference in the docs.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
API_BASE = "https://router.huggingface.co/hf-inference/models"
TIMEOUT_SECONDS = 30

# Heuristic, not tuned against real data: adjust after the first live run.
SEMANTIC_THRESHOLD = 0.80

MAX_REQUIRED = 40
MAX_RESUME_SKILLS = 60
MAX_SKILL_CHARS = 60

ALIASES = {
    "js": "javascript", "ecmascript": "javascript", "ts": "typescript",
    "node": "node.js", "nodejs": "node.js", "node js": "node.js",
    "reactjs": "react", "react.js": "react", "vuejs": "vue", "vue.js": "vue",
    "postgres": "postgresql", "k8s": "kubernetes", "ml": "machine learning",
    "amazon web services": "aws", "gcp": "google cloud", "google cloud platform": "google cloud",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "c sharp": "c#",
    "dotnet": ".net", "golang": "go", "cicd": "ci/cd", "ci cd": "ci/cd",
    "nlp": "natural language processing", "dl": "deep learning", "mongo": "mongodb",
}

# Curated vocabulary used to find required skills in free-text job descriptions.
SKILL_VOCABULARY = [
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "kotlin", "swift",
    "php", "ruby", "scala", "matlab", "bash", "sql", "html", "css", "react", "angular", "vue",
    "node.js", "express", "django", "flask", "fastapi", "spring boot", "spring", ".net", "next.js",
    "mysql", "postgresql", "postgres", "mongodb", "sqlite", "redis", "elasticsearch", "oracle",
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "ci/cd", "git", "github", "linux", "rest api", "graphql", "microservices", "kafka", "rabbitmq",
    "machine learning", "deep learning", "natural language processing", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark", "hadoop", "airflow",
    "tableau", "power bi", "excel", "data analysis", "data visualization", "statistics",
    "langchain", "langgraph", "llm", "streamlit", "huggingface", "opencv",
    "agile", "scrum", "jira", "testing", "pytest", "selenium", "unit testing",
    "communication", "leadership", "problem solving", "project management",
]

Embedder = Callable[[List[str], str, str], List[List[float]]]


class SkillApiError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "error": message, "data": None}


# ------------------------------------------------------------ normalization

def normalize_skill(skill: Any) -> str:
    text = re.sub(r"\s+", " ", str(skill).strip().lower())
    text = text[:MAX_SKILL_CHARS].strip(" ,;:-").rstrip(".")
    return ALIASES.get(text, text)


def _dedupe(skills: List[Any], limit: int) -> List[str]:
    seen, out = set(), []
    for s in skills:
        n = normalize_skill(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= limit:
            break
    return out


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def extract_required_skills(job_description: str) -> List[str]:
    text = job_description.lower()
    found = [t for t in SKILL_VOCABULARY if _contains_term(text, t)]
    return _dedupe(found, MAX_REQUIRED)


def _tokens(skill: str) -> set:
    return set(re.findall(r"[a-z0-9+#.]+", skill))


def _related(a: str, b: str) -> bool:
    """One skill's words are a strict subset of the other's (e.g. 'react' vs 'react native')."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and ta != tb and (ta < tb or tb < ta)


# ------------------------------------------------------------------ HF API

def _embed_hf(texts: List[str], token: str, model: str) -> List[List[float]]:
    """Single Hugging Face feature-extraction request. Isolated so tests can mock it."""
    url = f"{API_BASE}/{model}/pipeline/feature-extraction"
    body = json.dumps({"inputs": texts}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise SkillApiError("rate_limited", "Hugging Face rate limit reached. Try again later.")
        if exc.code in (401, 403):
            raise SkillApiError("auth_error", "Hugging Face rejected HF_TOKEN. Check the token and its Inference Providers permission.")
        if exc.code in (404, 503):
            raise SkillApiError("model_unavailable", f"Model \"{model}\" is unavailable on HF Inference. Set HF_MODEL in .env to another feature-extraction model.")
        raise SkillApiError("api_error", f"Hugging Face request failed (HTTP {exc.code}).")
    except (urllib.error.URLError, TimeoutError, OSError):
        raise SkillApiError("api_error", "Could not reach the Hugging Face API.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise SkillApiError("invalid_response", "Hugging Face returned an invalid (non-JSON) response.")
    return _validate_embeddings(parsed, len(texts))


def _validate_embeddings(parsed: Any, expected: int) -> List[List[float]]:
    ok = (
        isinstance(parsed, list)
        and len(parsed) == expected
        and all(
            isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
            for v in parsed
        )
        and len({len(v) for v in parsed}) == 1
    )
    if not ok:
        raise SkillApiError("invalid_response", "Hugging Face returned embeddings in an unexpected format.")
    return parsed


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _semantic_matches(
    unmatched: List[str],
    resume_skills: List[str],
    embedder: Embedder,
    token: str,
    model: str,
) -> Dict[str, Tuple[str, float]]:
    """Return {required_skill: (closest_resume_skill, similarity)} for pairs above threshold."""
    vectors = embedder(unmatched + resume_skills, token, model)
    req_vecs, res_vecs = vectors[: len(unmatched)], vectors[len(unmatched):]
    found = {}
    for skill, rv in zip(unmatched, req_vecs):
        best_skill, best_sim = max(
            ((rs, _cosine(rv, vv)) for rs, vv in zip(resume_skills, res_vecs)),
            key=lambda p: p[1],
        )
        if best_sim >= SEMANTIC_THRESHOLD:
            found[skill] = (best_skill, round(best_sim, 3))
    return found


# --------------------------------------------------------------------- core

def _recommendation(skill: str, priority: str) -> Dict[str, str]:
    if priority == "high":
        return {
            "skill": skill,
            "priority": "high",
            "reason": "Required by the target job but not found in your resume.",
            "action": f"Learn the basics of {skill} and build a small project using it; add it to your resume.",
        }
    return {
        "skill": skill,
        "priority": "medium",
        "reason": "Related experience found, but not an exact match.",
        "action": f"Strengthen {skill} with a focused project and list it explicitly on your resume.",
    }


def _required_from_job_match(job_match: Any) -> List[str]:
    try:
        top = job_match["data"]["matches"][0]["score_breakdown"]["skills"]
        if top.get("basis") == "job_required_skills":
            return list(top.get("matched", [])) + list(top.get("missing", []))
    except (KeyError, IndexError, TypeError):
        pass
    return []


def analyze_skill_gaps(
    resume: Optional[Dict[str, Any]],
    job_description: Optional[str] = None,
    required_skills: Optional[List[str]] = None,
    job_match: Optional[Dict[str, Any]] = None,
    embedder: Optional[Embedder] = None,
) -> Dict[str, Any]:
    """Compare resume skills with required skills. Never raises."""
    if not isinstance(resume, dict) or not resume:
        return _error("invalid_resume", "Parsed resume data is missing; run resume analysis first.")
    raw_skills = resume.get("skills")
    if raw_skills is not None and not isinstance(raw_skills, list):
        return _error("invalid_resume", "Resume 'skills' must be a list.")
    resume_skills = _dedupe(raw_skills or [], MAX_RESUME_SKILLS)

    source, candidates = None, []
    if isinstance(required_skills, list) and required_skills:
        source, candidates = "provided", required_skills
    elif isinstance(job_description, str) and job_description.strip():
        candidates = extract_required_skills(job_description)
        source = "job_description"
    if not candidates:
        candidates = _required_from_job_match(job_match)
        source = "job_match" if candidates else source
    required = _dedupe(candidates, MAX_REQUIRED)
    if not required:
        has_input = bool(required_skills) or (isinstance(job_description, str) and job_description.strip())
        return _error(
            "no_required_skills" if has_input or job_match else "empty_input",
            "Could not determine required skills from the job description."
            if has_input or job_match else "Provide a job description or required skills.",
        )

    matched, partial, unmatched = [], [], []
    resume_set = set(resume_skills)
    for skill in required:
        if skill in resume_set:
            matched.append({"skill": skill, "resume_skill": skill, "match_type": "exact"})
            continue
        rel = next((r for r in resume_skills if _related(skill, r)), None)
        if rel:
            partial.append({"skill": skill, "resume_skill": rel, "match_type": "related"})
        else:
            unmatched.append(skill)

    semantic: Dict[str, Any] = {"used": False, "model": None, "status": "skipped"}
    warning: Optional[Tuple[str, str]] = None
    if unmatched and resume_skills:
        load_dotenv()
        token = os.getenv("HF_TOKEN", "").strip()
        model = os.getenv("HF_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        semantic["model"] = model
        if not token or token.startswith(("your_", "hf_your")):
            warning = ("missing_api_token", "HF_TOKEN is not set; semantic comparison skipped. Add it to your .env file.")
        else:
            try:
                found = _semantic_matches(unmatched, resume_skills, embedder or _embed_hf, token, model)
                semantic.update(used=True, status="ok")
                for skill, (rs, sim) in found.items():
                    partial.append({"skill": skill, "resume_skill": rs, "match_type": "semantic", "similarity": sim})
                unmatched = [s for s in unmatched if s not in found]
            except SkillApiError as exc:
                logger.warning("Hugging Face call failed (%s).", exc.code)
                warning = (exc.code, exc.message)
            except Exception as exc:  # message omitted to avoid leaking request details
                logger.warning("Hugging Face call failed (%s).", type(exc).__name__)
                warning = ("api_error", "The Hugging Face request failed. Semantic comparison skipped.")
        if warning:
            semantic.update(status="unavailable", error_code=warning[0], error=warning[1])

    partial.sort(key=lambda p: required.index(p["skill"]))
    recommendations = [_recommendation(s, "high") for s in unmatched]
    recommendations += [_recommendation(p["skill"], "medium") for p in partial]
    coverage = round(100 * (len(matched) + 0.5 * len(partial)) / len(required))

    data = {
        "required_skills_source": source,
        "required_skills": required,
        "resume_skills": resume_skills,
        "matched": matched,
        "partial": partial,
        "missing": unmatched,
        "recommended_to_learn": recommendations,
        "coverage_percent": coverage,
        "semantic": semantic,
    }
    result: Dict[str, Any] = {"status": "partial" if warning else "ok", "data": data}
    if warning:
        result.update(error_code=warning[0], warning=warning[1])
    return result


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: reads resume_analysis, job_description (and job_match); writes skill_gaps."""
    analysis = state.get("resume_analysis")
    resume = analysis.get("data") if isinstance(analysis, dict) and analysis.get("status") == "ok" else None
    return {
        "skill_gaps": analyze_skill_gaps(
            resume,
            job_description=state.get("job_description"),
            required_skills=state.get("required_skills"),
            job_match=state.get("job_match"),
        )
    }