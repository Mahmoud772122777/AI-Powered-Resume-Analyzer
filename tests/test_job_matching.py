import json
import urllib.error

import pytest

from agents import job_matching as jm

RESUME = {
    "skills": ["Python", "SQL", "Docker"],
    "education": [{"institution": "MIT", "degree": "Bachelor of Science"}],
    "work_experience": [
        {"company": "A", "start_date": "2019", "end_date": "2021"},
        {"company": "B", "start_date": "2021", "end_date": "2023"},
    ],
}

JOB_GOOD = {
    "job_id": "1",
    "job_title": "Python Developer",
    "employer_name": "Acme",
    "job_city": "Pune",
    "job_country": "IN",
    "job_apply_link": "https://example.com/1",
    "job_description": "Build services.",
    "job_required_skills": ["Python", "SQL"],
    "job_required_experience": {"required_experience_in_months": 36},
    "job_required_education": {"bachelors_degree": True},
}
JOB_POOR = {
    "job_id": "2",
    "job_title": "Rust Engineer",
    "job_description": "Rust and Kubernetes.",
    "job_required_skills": ["Rust", "Kubernetes"],
    "job_required_experience": {"required_experience_in_months": 120},
}


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "test-key-123")
    monkeypatch.setattr(jm, "load_dotenv", lambda *a, **k: None)


def fetcher_returning(payload):
    calls = []

    def fetcher(params, key):
        calls.append((params, key))
        if isinstance(payload, Exception):
            raise payload
        return payload

    fetcher.calls = calls
    return fetcher


def test_scores_and_ranks_with_breakdown():
    f = fetcher_returning({"status": "OK", "data": [JOB_POOR, JOB_GOOD]})
    out = jm.match_jobs(RESUME, target_role="Python Developer", fetcher=f)
    assert out["status"] == "ok"
    matches = out["data"]["matches"]
    assert [m["title"] for m in matches] == ["Python Developer", "Rust Engineer"]
    best = matches[0]
    assert best["match_score"] == 100
    bd = best["score_breakdown"]
    assert bd["skills"]["matched"] == ["python", "sql"]
    assert bd["experience"]["resume_months"] == 48
    assert bd["qualifications"]["score"] == 100
    assert matches[1]["score_breakdown"]["skills"]["missing"] == ["rust", "kubernetes"]
    assert matches[1]["match_score"] < 20
    assert len(f.calls) == 1
    json.dumps(out)


def test_description_fallback_when_no_required_skills():
    job = {"job_title": "Dev", "job_description": "We use Python and Docker daily."}
    res = jm.score_job(RESUME, job)
    s = res["score_breakdown"]["skills"]
    assert s["basis"] == "skills_found_in_description"
    assert s["matched"] == ["python", "docker"]
    assert res["match_score"] == 67


def test_unknown_components_are_excluded_not_zeroed():
    job = {"job_required_skills": ["Python"]}
    assert jm.score_job(RESUME, job)["match_score"] == 100


def test_partial_experience_and_education():
    resume = {"skills": [], "education": [{"degree": "High School"}],
              "work_experience": [{"start_date": "2020", "end_date": "2021"}]}
    job = {"job_required_experience": {"required_experience_in_months": 24},
           "job_required_education": {"bachelors_degree": True}}
    bd = jm.score_job(resume, job)["score_breakdown"]
    assert bd["experience"]["score"] == 50
    assert bd["qualifications"]["score"] == 33


def test_query_uses_job_description_first_line():
    f = fetcher_returning({"data": []})
    jm.match_jobs(RESUME, job_description="Data Analyst\nLong text", fetcher=f)
    assert f.calls[0][0]["query"] == "Data Analyst"


def test_search_request_does_not_include_resume_or_personal_data():
    private_resume = {
        **RESUME,
        "name": "Jane Secret",
        "contact": {"email": "jane@secret.example", "phone": "555-123-4567"},
    }
    fetcher = fetcher_returning({"data": []})

    jm.match_jobs(
        private_resume,
        target_role="Python Developer",
        job_description="Python Developer\nBuild services.",
        fetcher=fetcher,
    )

    params = fetcher.calls[0][0]
    assert params == {"query": "Python Developer", "page": "1", "num_pages": "1"}
    assert "Jane" not in json.dumps(params)
    assert "secret.example" not in json.dumps(params)


def test_empty_results():
    out = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning({"data": []}))
    assert out["status"] == "ok"
    assert out["data"]["matches"] == [] and out["data"]["message"]


def test_missing_key_makes_no_call(monkeypatch):
    monkeypatch.delenv("JSEARCH_API_KEY")
    f = fetcher_returning({"data": []})
    assert jm.match_jobs(RESUME, target_role="x", fetcher=f)["error_code"] == "missing_api_key"
    assert f.calls == []


def test_invalid_resume_and_empty_target():
    f = fetcher_returning({"data": []})
    assert jm.match_jobs(None, target_role="x", fetcher=f)["error_code"] == "invalid_resume"
    assert jm.match_jobs(RESUME, fetcher=f)["error_code"] == "empty_input"
    assert f.calls == []


@pytest.mark.parametrize("exc,code", [
    (jm.JobApiError("rate_limited", "limit"), "rate_limited"),
    (jm.JobApiError("auth_error", "bad key"), "auth_error"),
    (RuntimeError("boom test-key-123"), "api_error"),
])
def test_api_errors(exc, code):
    out = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning(exc))
    assert out["error_code"] == code
    assert "test-key-123" not in json.dumps(out)


def test_invalid_response_shape():
    out = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning({"data": "nope"}))
    assert out["error_code"] == "invalid_response"


def test_v2_data_dict_with_jobs():
    payload = {"status": "OK", "data": {"jobs": [JOB_POOR, JOB_GOOD], "cursor": "abc"}}
    legacy = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning(
        {"data": [JOB_POOR, JOB_GOOD]}))
    out = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning(payload))
    assert out["status"] == "ok"
    assert out["data"]["matches"] == legacy["data"]["matches"]
    assert len(out["data"]["matches"]) == 2


def test_v2_empty_jobs():
    out = jm.match_jobs(RESUME, target_role="x",
                        fetcher=fetcher_returning({"data": {"jobs": [], "cursor": ""}}))
    assert out["status"] == "ok"
    assert out["data"]["matches"] == [] and out["data"]["message"]


@pytest.mark.parametrize("payload", [
    {"data": {"cursor": "x"}},
    {"data": {"jobs": "nope"}},
    {"data": None},
    {"status": "OK"},
    ["not", "a", "dict"],
])
def test_v2_malformed_responses(payload):
    out = jm.match_jobs(RESUME, target_role="x", fetcher=fetcher_returning(payload))
    assert out["status"] == "error" and out["error_code"] == "invalid_response"


def test_http_429_maps_to_rate_limited(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)

    monkeypatch.setattr(jm.urllib.request, "urlopen", boom)
    with pytest.raises(jm.JobApiError) as e:
        jm._fetch_jsearch({"query": "x"}, "test-key-123")
    assert e.value.code == "rate_limited"
    assert "test-key-123" not in e.value.message


def test_run_node_reads_resume_analysis(monkeypatch):
    monkeypatch.setattr(jm, "_fetch_jsearch", lambda p, k: {"data": [JOB_GOOD]})
    state = {"resume_analysis": {"status": "ok", "data": RESUME}, "job_description": "Python Developer"}
    out = jm.run(state)
    assert out["job_match"]["data"]["matches"][0]["match_score"] == 100


def test_run_node_without_valid_resume():
    out = jm.run({"resume_analysis": {"status": "error"}, "job_description": "x"})
    assert out["job_match"]["error_code"] == "invalid_resume"