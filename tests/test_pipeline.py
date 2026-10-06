from agents import career_guidance, job_matching, resume_analysis, skill_gap_analysis
from orchestration import run_pipeline


def test_pipeline_runs_all_agents(monkeypatch):
    monkeypatch.setattr(
        resume_analysis,
        "analyze_resume",
        lambda text: {"status": "ok", "data": {"name": "Test"}},
    )
    monkeypatch.setattr(
        job_matching,
        "match_jobs",
        lambda resume, target_role=None, job_description=None: {"status": "ok", "data": {"matches": []}},
    )
    monkeypatch.setattr(
        skill_gap_analysis,
        "analyze_skill_gaps",
        lambda *a, **k: {"status": "ok", "data": {"missing": []}},
    )
    monkeypatch.setattr(
        career_guidance,
        "generate_guidance",
        lambda **k: {"status": "ok", "data": {}},
    )
    result = run_pipeline("sample resume", "sample job description")
    assert result["resume_analysis"]["status"] == "ok"
    assert result["job_match"]["status"] == "ok"
    assert result["skill_gaps"]["status"] == "ok"
    assert result["career_guidance"]["status"] == "ok"