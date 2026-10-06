from orchestration import run_pipeline


def test_pipeline_runs_all_placeholder_agents():
    result = run_pipeline("sample resume", "sample job description")
    for key in ("resume_analysis", "job_match", "skill_gaps", "career_guidance"):
        assert result[key]["status"] == "placeholder"