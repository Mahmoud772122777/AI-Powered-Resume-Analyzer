"""LangGraph orchestration for the four agents.

LangGraph is the coordinator, not an agent. The four agents (Gemini, JSearch,
Hugging Face, Apify) keep their own responsibilities. This module adds:

* conditional routing (stop early if resume analysis fails),
* a deterministic cross-agent critique node (plain Python rules, no API calls),
* one bounded revision (re-runs only Job Matching and Skill Gap Analysis),
* a trace that records only what actually executed or was decided.
"""
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    career_guidance,
    job_matching,
    resume_analysis,
    skill_gap_analysis,
)

HARD_MAX_ITERATIONS = 2  # initial attempt + at most one revision
LOW_COVERAGE = 40
HIGH_MATCH = 70


class AnalyzerState(TypedDict, total=False):
    # inputs
    resume_text: str
    job_description: str
    target_role: Optional[str]
    country: Optional[str]
    required_skills: Optional[List[str]]
    # agent outputs
    resume_analysis: Dict[str, Any]
    job_match: Dict[str, Any]
    skill_gaps: Dict[str, Any]
    career_guidance: Dict[str, Any]
    # orchestration control / observability
    iteration: int
    max_iterations: int
    next_step: str
    critique: Dict[str, Any]
    revision_history: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    final_outcome: Dict[str, Any]


# ------------------------------------------------------------------ helpers

def _trace(state: Dict[str, Any], **entry: Any) -> List[Dict[str, Any]]:
    trace = list(state.get("trace") or [])
    trace.append({"step": len(trace) + 1, **entry})
    return trace


def _status(result: Any) -> str:
    return result.get("status", "unknown") if isinstance(result, dict) else "missing"


def _code(result: Any) -> Optional[str]:
    return result.get("error_code") if isinstance(result, dict) else None


def _matches(job_match: Any) -> List[Dict[str, Any]]:
    data = job_match.get("data") if isinstance(job_match, dict) else None
    found = data.get("matches") if isinstance(data, dict) else None
    return [m for m in found if isinstance(m, dict)] if isinstance(found, list) else []


def _executed(state: Dict[str, Any], node: str, result: Any, **extra: Any) -> List[Dict[str, Any]]:
    return _trace(state, node=node, event="executed", status=_status(result),
                  error_code=_code(result), **extra)


def _alternative_query(state: Dict[str, Any]) -> Optional[str]:
    """Use a job-description title only when its terms form a strict subset of the role."""
    role = state.get("target_role")
    description = state.get("job_description")
    if not (isinstance(role, str) and role.strip()):
        return None
    if not (isinstance(description, str) and description.strip()):
        return None
    alt = job_matching.build_query(None, description)
    original = job_matching.build_query(role, None)
    role_terms = set(re.findall(r"[a-z0-9+#.]+", original.lower()))
    alternative_terms = set(re.findall(r"[a-z0-9+#.]+", alt.lower()))
    if alt and alternative_terms and alternative_terms < role_terms:
        return alt
    return None


# -------------------------------------------------------------------- nodes

def resume_node(state: AnalyzerState) -> Dict[str, Any]:
    update = resume_analysis.run(state)
    result = update.get("resume_analysis")
    ok = _status(result) == "ok"
    nxt = "job_matching" if ok else "finalize"
    trace = _executed(state, "resume_analysis", result)
    trace.append({
        "step": len(trace) + 1, "node": "router", "event": "decision",
        "decision": nxt,
        "reason": "Resume analysis succeeded." if ok else
        "Resume analysis did not succeed; downstream agents need parsed resume data, so the run stops.",
    })
    return {**update, "next_step": nxt, "trace": trace}


def job_matching_node(state: AnalyzerState) -> Dict[str, Any]:
    update = job_matching.run(state)
    result = update.get("job_match")
    return {**update, "trace": _executed(
        state, "job_matching", result, matches=len(_matches(result)))}


def skill_gap_node(state: AnalyzerState) -> Dict[str, Any]:
    update = skill_gap_analysis.run(state)
    result = update.get("skill_gaps")
    return {**update, "trace": _executed(state, "skill_gap_analysis", result)}


def critique_node(state: AnalyzerState) -> Dict[str, Any]:
    """Deterministic cross-agent checks. May request one revision."""
    iteration = state.get("iteration", 1)
    cap = min(state.get("max_iterations", HARD_MAX_ITERATIONS), HARD_MAX_ITERATIONS)
    job_match = state.get("job_match")
    gaps = state.get("skill_gaps")
    resume = (state.get("resume_analysis") or {}).get("data") or {}
    issues: List[Dict[str, Any]] = []
    revision: Optional[Dict[str, Any]] = None

    if _status(job_match) == "ok" and not _matches(job_match):
        alt = _alternative_query(state)
        recoverable = alt is not None and iteration < cap
        issues.append({
            "code": "job_search_empty", "severity": "warning", "recoverable": recoverable,
            "message": "The job search returned no listings."
            + (" A broader query derived from the job description is available." if alt else
               " No different query can be derived from the supplied inputs."),
        })
        if recoverable:
            revision = {"reason": "job_search_empty", "action": "rerun_job_matching_and_skill_gap",
                        "revised_query": alt}
    elif _status(job_match) != "ok":
        issues.append({
            "code": "job_match_unavailable", "severity": "warning", "recoverable": False,
            "error_code": _code(job_match),
            "message": "Job matching did not succeed (not retried); later guidance relies on the other agents.",
        })

    if not resume.get("skills"):
        issues.append({"code": "resume_no_skills", "severity": "warning", "recoverable": False,
                       "message": "No skills were extracted from the resume, so skill comparisons are limited."})

    if _status(gaps) == "partial":
        issues.append({"code": "semantic_analysis_unavailable", "severity": "info", "recoverable": False,
                       "error_code": _code(gaps),
                       "message": "Semantic skill comparison was unavailable; results use deterministic matching only."})
    elif _status(gaps) == "error":
        issues.append({"code": "skill_gap_unavailable", "severity": "warning", "recoverable": False,
                       "error_code": _code(gaps),
                       "message": "Skill gap analysis did not produce results."})
    else:
        matches = _matches(job_match)
        coverage = (gaps.get("data") or {}).get("coverage_percent") if isinstance(gaps, dict) else None
        top = matches[0].get("match_score") if matches else None
        if matches and isinstance(coverage, (int, float)) and coverage < LOW_COVERAGE:
            alt = _alternative_query(state)
            recoverable = alt is not None and iteration < cap and revision is None
            issues.append({
                "code": "low_skill_coverage", "severity": "warning", "recoverable": recoverable,
                "message": f"Skill coverage is only {coverage}% (below {LOW_COVERAGE}%)."
                + (" A broader query derived from the job description will be tried."
                   if recoverable else " No further retry is possible."),
            })
            if recoverable:
                revision = {"reason": "low_skill_coverage",
                            "action": "rerun_job_matching_and_skill_gap",
                            "revised_query": alt}
        if isinstance(top, (int, float)) and isinstance(coverage, (int, float)) \
                and top >= HIGH_MATCH and coverage < LOW_COVERAGE:
            issues.append({"code": "score_coverage_mismatch", "severity": "warning", "recoverable": False,
                           "message": f"Top job match is {top}% but skill coverage is {coverage}%; "
                                      "treat the match score with caution."})

    nxt = "revise" if revision else "career_guidance"
    critique = {"iteration": iteration, "issues": issues, "revision_requested": bool(revision),
                "unresolved": [i["code"] for i in issues if i["severity"] != "info"]}
    if revision:
        critique["revision"] = revision
    trace = _trace(state, node="critique", event="executed", issues=[i["code"] for i in issues])
    trace.append({
        "step": len(trace) + 1, "node": "router", "event": "decision", "decision": nxt,
        "reason": f"Revision requested: {revision['reason']}." if revision else
        ("No recoverable issue found." if iteration < cap or not issues else
         "Iteration cap reached; remaining issues are reported, not retried."),
    })
    return {"critique": critique, "next_step": nxt, "trace": trace}


def revise_node(state: AnalyzerState) -> Dict[str, Any]:
    """Re-run Job Matching with a broader query and Skill Gap Analysis only."""
    revision = state["critique"]["revision"]
    original_query = ((state.get("job_match") or {}).get("data") or {}).get("query")
    trace = _trace(
        state,
        node="revise",
        event="executed",
        reason=revision["reason"],
        candidate_query=revision["revised_query"],
    )
    # target_role is cleared only for this call so the query comes from the job description.
    revised = job_matching.run({**state, "target_role": None})["job_match"]
    trace.append({
        "step": len(trace) + 1, "node": "job_matching", "event": "executed",
        "revision": True, "status": _status(revised), "error_code": _code(revised),
        "matches": len(_matches(revised)),
    })
    update: Dict[str, Any] = {"iteration": state.get("iteration", 1) + 1}
    adopted = _status(revised) == "ok" and bool(_matches(revised))
    gaps = skill_gap_analysis.run({**state, "job_match": revised})["skill_gaps"]
    if revision["reason"] == "low_skill_coverage":
        # Evidence-based decision: keep the revised results only if coverage improved.
        old_cov = ((state.get("skill_gaps") or {}).get("data") or {}).get("coverage_percent")
        new_cov = ((gaps or {}).get("data") or {}).get("coverage_percent") \
            if isinstance(gaps, dict) else None
        adopted = adopted and isinstance(old_cov, (int, float)) \
            and isinstance(new_cov, (int, float)) and new_cov > old_cov
        if adopted:
            update["skill_gaps"] = gaps
    else:
        update["skill_gaps"] = gaps
    if adopted:
        update["job_match"] = revised
        outcome = "revised_results_adopted"
    else:
        outcome = "revised_search_did_not_help"
    trace.append({
        "step": len(trace) + 1, "node": "skill_gap_analysis", "event": "executed",
        "revision": True, "status": _status(gaps), "error_code": _code(gaps),
    })
    history = list(state.get("revision_history") or [])
    history.append({
        "iteration": update["iteration"], "reason": revision["reason"],
        "original_query": original_query, "revised_query": revision["revised_query"],
        "outcome": outcome, "result_status": _status(revised),
        "error_code": _code(revised),
    })
    update["revision_history"] = history
    update["trace"] = trace
    return update


def career_guidance_node(state: AnalyzerState) -> Dict[str, Any]:
    update = career_guidance.run(state)
    result = update.get("career_guidance")
    source = (result or {}).get("data", {}).get("guidance_source") if isinstance(result, dict) \
        and isinstance(result.get("data"), dict) else None
    return {**update, "trace": _executed(state, "career_guidance", result, guidance_source=source)}


def finalize_node(state: AnalyzerState) -> Dict[str, Any]:
    statuses = {
        "resume_analysis": _status(state.get("resume_analysis")),
        "job_matching": _status(state.get("job_match")) if "job_match" in state else "not_run",
        "skill_gap_analysis": _status(state.get("skill_gaps")) if "skill_gaps" in state else "not_run",
        "career_guidance": _status(state.get("career_guidance")) if "career_guidance" in state else "not_run",
    }
    revised = bool(state.get("revision_history"))
    if statuses["resume_analysis"] != "ok":
        label = "stopped_resume_analysis_failed"
    elif any(s != "ok" for s in statuses.values()):
        label = "completed_with_warnings"
    else:
        label = "completed"
    outcome = {"outcome": label, "revised": revised, "agent_statuses": statuses}
    trace = _trace(
        state, node="finalize", event="final", outcome=label, revised=revised,
        agent_statuses=statuses,
    )
    return {"final_outcome": outcome, "trace": trace}


# ----------------------------------------------------------------- the graph

def build_graph():
    """Build and compile the workflow graph."""
    graph = StateGraph(AnalyzerState)
    graph.add_node("resume_analysis", resume_node)
    graph.add_node("job_matching", job_matching_node)
    graph.add_node("skill_gap_analysis", skill_gap_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_node("career_guidance", career_guidance_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("resume_analysis")
    graph.add_conditional_edges(
        "resume_analysis", lambda s: s["next_step"],
        {"job_matching": "job_matching", "finalize": "finalize"})
    graph.add_edge("job_matching", "skill_gap_analysis")
    graph.add_edge("skill_gap_analysis", "critique")
    graph.add_conditional_edges(
        "critique", lambda s: s["next_step"],
        {"revise": "revise", "career_guidance": "career_guidance"})
    graph.add_edge("revise", "critique")
    graph.add_edge("career_guidance", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _clean(value: Optional[str]) -> Optional[str]:
    return value.strip() or None if isinstance(value, str) else None


def run_pipeline(
    resume_text: str,
    job_description: str,
    target_role: Optional[str] = None,
    country: Optional[str] = None,
    required_skills: Optional[List[str]] = None,
    max_iterations: int = HARD_MAX_ITERATIONS,
) -> Optional[Dict[str, Any]]:
    """Run the full pipeline and return the final state."""
    app = build_graph()
    initial: Dict[str, Any] = {
        "resume_text": resume_text,
        "job_description": job_description,
        "target_role": _clean(target_role),
        "country": _clean(country),
        "required_skills": required_skills or None,
        "iteration": 1,
        "max_iterations": max(1, min(max_iterations, HARD_MAX_ITERATIONS)),
        "revision_history": [],
        "trace": [],
    }
    return app.invoke(initial)