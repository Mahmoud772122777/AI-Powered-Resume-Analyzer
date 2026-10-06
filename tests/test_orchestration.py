import pytest

from agents import career_guidance, job_matching, resume_analysis, skill_gap_analysis
from orchestration import run_pipeline


def _ok(data):
    return {"status": "ok", "data": data}


def _match(query, matches=None):
    return _ok({"query": query, "matches": matches or []})


def test_pipeline_passes_explicit_inputs_and_traces_nodes(monkeypatch):
    calls = {"resume": [], "jobs": [], "gaps": [], "guidance": []}
    resume = {"skills": ["Python"]}

    def run_resume(state):
        calls["resume"].append(state)
        return {"resume_analysis": _ok(resume)}

    def run_jobs(state):
        calls["jobs"].append(state)
        return {"job_match": _match(state["target_role"], [{"title": "Data Engineer", "match_score": 80}])}

    def run_gaps(state):
        calls["gaps"].append(state)
        return {"skill_gaps": _ok({
            "required_skills": state["required_skills"], "coverage_percent": 80,
        })}

    def run_guidance(state):
        calls["guidance"].append(state)
        return {"career_guidance": _ok({"guidance_source": "fallback_no_market_data"})}

    monkeypatch.setattr(resume_analysis, "run", run_resume)
    monkeypatch.setattr(job_matching, "run", run_jobs)
    monkeypatch.setattr(skill_gap_analysis, "run", run_gaps)
    monkeypatch.setattr(career_guidance, "run", run_guidance)

    result = run_pipeline(
        "private resume text",
        "Data Engineer\nBuild data systems.",
        target_role="Data Engineer",
        country="IN",
        required_skills=["Python", "SQL"],
    )

    assert len(calls["resume"]) == 1
    assert calls["jobs"][0]["target_role"] == "Data Engineer"
    assert calls["gaps"][0]["required_skills"] == ["Python", "SQL"]
    assert calls["guidance"][0]["target_role"] == "Data Engineer"
    assert calls["guidance"][0]["country"] == "IN"
    assert result["final_outcome"]["outcome"] == "completed"
    assert result["final_outcome"]["agent_statuses"]["career_guidance"] == "ok"
    assert [entry["node"] for entry in result["trace"] if entry["event"] == "executed"] == [
        "resume_analysis", "job_matching", "skill_gap_analysis", "critique", "career_guidance",
    ]
    assert result["trace"][-1]["event"] == "final"


def test_revision_retries_only_matching_and_skill_gaps_once(monkeypatch):
    calls = {"resume": 0, "jobs": [], "gaps": [], "guidance": []}

    def run_resume(state):
        calls["resume"] += 1
        return {"resume_analysis": _ok({"skills": ["Python"]})}

    def run_jobs(state):
        calls["jobs"].append(state)
        query = job_matching.build_query(state.get("target_role"), state["job_description"])
        matches = [] if len(calls["jobs"]) == 1 else [{"title": "Data Engineer", "match_score": 75}]
        return {"job_match": _match(query, matches)}

    def run_gaps(state):
        calls["gaps"].append(state)
        return {"skill_gaps": _ok({
            "required_skills": state["required_skills"], "coverage_percent": 60,
        })}

    def run_guidance(state):
        calls["guidance"].append(state)
        return {"career_guidance": _ok({"guidance_source": "fallback_no_market_data"})}

    monkeypatch.setattr(resume_analysis, "run", run_resume)
    monkeypatch.setattr(job_matching, "run", run_jobs)
    monkeypatch.setattr(skill_gap_analysis, "run", run_gaps)
    monkeypatch.setattr(career_guidance, "run", run_guidance)

    result = run_pipeline(
        "private resume text",
        "Data Engineer\nBuild data systems.",
        target_role="Senior Data Engineer",
        country="IN",
        required_skills=["Python", "SQL"],
        max_iterations=20,
    )

    assert calls["resume"] == 1
    assert len(calls["jobs"]) == len(calls["gaps"]) == 2
    assert calls["jobs"][0]["target_role"] == "Senior Data Engineer"
    assert calls["jobs"][1]["target_role"] is None
    assert calls["jobs"][1]["country"] == "IN"
    assert all(call["required_skills"] == ["Python", "SQL"] for call in calls["gaps"])
    assert calls["guidance"][0]["target_role"] == "Senior Data Engineer"
    assert calls["guidance"][0]["country"] == "IN"
    assert result["iteration"] == 2
    assert result["revision_history"] == [{
        "iteration": 2,
        "reason": "job_search_empty",
        "original_query": "Senior Data Engineer",
        "revised_query": "Data Engineer",
        "outcome": "revised_results_adopted",
        "result_status": "ok",
        "error_code": None,
    }]
    trace = result["trace"]
    assert [entry["node"] for entry in trace if entry["event"] == "executed"] == [
        "resume_analysis", "job_matching", "skill_gap_analysis", "critique",
        "revise", "job_matching", "skill_gap_analysis", "critique", "career_guidance",
    ]
    assert sum(entry["event"] == "decision" and entry.get("decision") == "revise"
               for entry in trace) == 1
    assert trace[-1]["event"] == "final"


def test_revision_with_no_improvement_still_stops_at_two_attempts(monkeypatch):
    job_calls = []
    gap_calls = []
    monkeypatch.setattr(
        resume_analysis, "run",
        lambda state: {"resume_analysis": _ok({"skills": ["Python"]})},
    )

    def run_jobs(state):
        job_calls.append(state)
        return {"job_match": _match(
            job_matching.build_query(state.get("target_role"), state["job_description"]),
        )}

    def run_gaps(state):
        gap_calls.append(state)
        return {"skill_gaps": _ok({"required_skills": ["Python"], "coverage_percent": 100})}

    monkeypatch.setattr(job_matching, "run", run_jobs)
    monkeypatch.setattr(skill_gap_analysis, "run", run_gaps)
    monkeypatch.setattr(
        career_guidance, "run",
        lambda state: {"career_guidance": _ok({"guidance_source": "fallback_no_market_data"})},
    )

    result = run_pipeline(
        "private resume text", "Data Engineer\nBuild data systems.",
        target_role="Senior Data Engineer", max_iterations=100,
    )

    assert len(job_calls) == len(gap_calls) == 2
    assert result["iteration"] == 2
    assert result["revision_history"][0]["outcome"] == "revised_search_did_not_help"
    assert result["trace"][-1]["event"] == "final"


def test_revision_is_not_attempted_for_a_narrower_or_unrelated_query(monkeypatch):
    job_calls = []
    monkeypatch.setattr(
        resume_analysis, "run",
        lambda _state: {"resume_analysis": _ok({"skills": ["Python"]})},
    )

    def run_jobs(state):
        job_calls.append(state)
        return {"job_match": _match(
            job_matching.build_query(state.get("target_role"), state["job_description"]),
        )}

    monkeypatch.setattr(job_matching, "run", run_jobs)
    monkeypatch.setattr(
        skill_gap_analysis, "run",
        lambda _state: {"skill_gaps": _ok({"required_skills": ["Python"], "coverage_percent": 100})},
    )
    monkeypatch.setattr(
        career_guidance, "run",
        lambda _state: {"career_guidance": _ok({"guidance_source": "fallback_no_market_data"})},
    )

    result = run_pipeline(
        "private resume text", "Senior Data Engineer\nBuild data systems.",
        target_role="Data Engineer",
    )

    assert len(job_calls) == 1
    assert result["revision_history"] == []
    assert not any(entry["node"] == "revise" for entry in result["trace"])


@pytest.mark.parametrize("error_code", ["auth_error", "rate_limited", "api_error"])
def test_nonrecoverable_job_api_errors_are_not_retried(monkeypatch, error_code):
    jobs = []
    monkeypatch.setattr(
        resume_analysis, "run",
        lambda state: {"resume_analysis": _ok({"skills": ["Python"]})},
    )

    def run_jobs(state):
        jobs.append(state)
        return {"job_match": {
            "status": "error", "error_code": error_code, "data": None,
        }}

    monkeypatch.setattr(job_matching, "run", run_jobs)
    monkeypatch.setattr(
        skill_gap_analysis, "run",
        lambda state: {"skill_gaps": _ok({"required_skills": ["Python"], "coverage_percent": 100})},
    )
    monkeypatch.setattr(
        career_guidance, "run",
        lambda state: {"career_guidance": _ok({"guidance_source": "fallback_no_market_data"})},
    )

    result = run_pipeline(
        "private resume text", "Data Engineer\nBuild data systems.",
        target_role="Senior Data Engineer",
    )

    assert len(jobs) == 1
    assert result["revision_history"] == []
    critique = next(entry for entry in result["trace"] if entry["node"] == "critique")
    assert critique["issues"][0] == "job_match_unavailable"
    assert not any(entry["node"] == "revise" for entry in result["trace"])


def test_resume_analysis_failure_stops_downstream_agents(monkeypatch):
    calls = []
    monkeypatch.setattr(
        resume_analysis, "run",
        lambda state: {"resume_analysis": {"status": "error", "error_code": "api_error"}},
    )
    monkeypatch.setattr(job_matching, "run", lambda state: calls.append("jobs"))
    monkeypatch.setattr(skill_gap_analysis, "run", lambda state: calls.append("gaps"))
    monkeypatch.setattr(career_guidance, "run", lambda state: calls.append("guidance"))

    result = run_pipeline("private resume text", "Data Engineer")

    assert calls == []
    assert result["final_outcome"]["outcome"] == "stopped_resume_analysis_failed"
    assert result["final_outcome"]["agent_statuses"]["job_matching"] == "not_run"
