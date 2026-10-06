"""All Apify data below is MOCK test data; no network calls are made."""
import io
import json
import socket
import urllib.error

import pytest

from agents import career_guidance as cg

RESUME = {"status": "ok", "data": {
    "name": "Jane Secret", "contact": {"email": "jane@secret.com", "phone": "555-123-4567"},
    "skills": ["Python", "SQL"], "projects": [], "certifications": [], "work_experience": [],
    "professional_summary": None}}
MATCH = {"status": "ok", "data": {"matches": [
    {"title": "Data Analyst", "company": "Acme", "match_score": 55, "apply_link": "https://x.test/1"}]}}
GAPS = {"status": "ok", "data": {
    "matched": [{"skill": "python"}], "missing": ["tableau"], "coverage_percent": 50,
    "recommended_to_learn": [{"skill": "tableau", "priority": "high", "reason": "Required.", "action": "Learn"}]}}
MOCK_ITEMS = [
    {"positionName": "Data Analyst", "company": "Mock Co", "location": "Remote", "salary": "$80k",
     "jobType": ["Full-time"], "url": "https://example.test/j1", "postedAt": "Today",
     "description": "Python, SQL and Tableau required. Docker nice."},
    {"positionName": "BI Analyst", "company": "Mock 2", "url": "javascript:bad",
     "description": "Tableau and Python."},
]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "apify_api_secret-token-xyz")
    monkeypatch.setattr(cg, "load_dotenv", lambda *a, **k: None)


def recorder(items=None, error=None):
    calls = []

    def runner(token, actor_input):
        calls.append(actor_input)
        if error:
            raise error
        return MOCK_ITEMS if items is None else items

    runner.calls = calls
    return runner


def gen(**kw):
    kw.setdefault("resume_analysis", RESUME)
    kw.setdefault("job_match", MATCH)
    kw.setdefault("skill_gaps", GAPS)
    return cg.generate_guidance(**kw)


def test_live_market_guidance():
    r = recorder()
    out = gen(target_role="Data Analyst", runner=r)
    d = out["data"]
    assert out["status"] == "ok" and d["guidance_source"] == "apify_live"
    assert d["market"]["source"] == "apify_live" and d["market"]["listing_count"] == 2
    top = {s["skill"]: s for s in d["market"]["top_skills"]}
    assert top["tableau"]["listings"] == 2 and top["tableau"]["percent"] == 100
    assert top["python"]["in_resume"] is True
    assert d["market"]["sample_listings"][1]["url"] is None  # unsafe URL dropped
    assert d["skills_to_prioritize"][0]["skill"] == "tableau"
    assert d["skills_to_prioritize"][0]["evidence"] == "skill_gap_analysis"
    assert d["project_ideas"] and d["job_search_suggestions"] and d["resume_improvements"]
    json.dumps(out)


def test_apify_input_has_no_personal_data():
    r = recorder()
    gen(target_role="Data Analyst jane@secret.com 5551234567", country="gb", runner=r)
    sent = json.dumps(r.calls[0])
    assert r.calls[0] == {"position": "Data Analyst", "maxItemsPerSearch": cg.MAX_ITEMS, "country": "GB"}
    assert "secret" not in sent.lower() and "Jane" not in sent


def test_role_falls_back_to_top_job_match_title():
    r = recorder()
    out = gen(runner=r)
    assert out["data"]["target_role"] == "Data Analyst" and r.calls[0]["position"] == "Data Analyst"


def test_invalid_country_ignored():
    r = recorder()
    gen(target_role="Dev", country="USA; drop", runner=r)
    assert "country" not in r.calls[0]


def test_missing_token_gives_fallback_without_call(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN")
    monkeypatch.setattr(cg, "_run_actor", lambda *a: pytest.fail("must not call Apify"))
    out = gen(target_role="Data Analyst")
    assert out["status"] == "partial" and out["error_code"] == "missing_api_token"
    assert out["data"]["guidance_source"] == "fallback_no_market_data"
    assert out["data"]["market"] == {"source": "unavailable"}
    assert out["data"]["skills_to_prioritize"][0]["skill"] == "tableau"


def test_placeholder_token_treated_as_missing(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "your_apify_api_token_here")
    assert gen(target_role="x")["error_code"] == "missing_api_token"


@pytest.mark.parametrize("code", ["rate_limited", "auth_error", "quota_exceeded", "actor_unavailable",
                                  "timeout", "actor_error", "invalid_response", "api_error"])
def test_apify_errors_fall_back(code):
    out = gen(target_role="Data Analyst", runner=recorder(error=cg.ApifyError(code, "m")))
    assert out["status"] == "partial" and out["error_code"] == code
    assert out["data"]["guidance_source"] == "fallback_no_market_data"
    assert out["data"]["project_ideas"]


def test_unexpected_exception_does_not_leak_token():
    out = gen(target_role="x", runner=recorder(error=RuntimeError("boom apify_api_secret-token-xyz")))
    assert out["error_code"] == "api_error"
    assert "secret-token" not in json.dumps(out)


def test_empty_results_fall_back():
    assert gen(target_role="x", runner=recorder(items=[]))["error_code"] == "no_results"


def test_non_list_runner_result():
    out = gen(target_role="x", runner=lambda t, i: {"bad": 1})
    assert out["error_code"] == "invalid_response"


def test_no_role_skips_apify():
    r = recorder()
    out = gen(job_match=None, runner=r)
    assert out["error_code"] == "no_search_term" and r.calls == []
    assert out["status"] == "partial"


def test_no_input_is_error():
    out = cg.generate_guidance(resume_analysis={"status": "error"}, job_match=None, skill_gaps=None)
    assert out["status"] == "error" and out["error_code"] == "no_input"


def test_resume_improvements_and_search_tips():
    out = gen(target_role="x", runner=recorder())["data"]
    text = " ".join(out["resume_improvements"])
    assert "summary" in text and "projects" in text and "certifications" in text
    assert any("55/100" in s for s in out["job_search_suggestions"])


def test_sanitize_role():
    assert cg.sanitize_role("  Senior C++ Dev  ") == "Senior C++ Dev"
    assert cg.sanitize_role("") is None and cg.sanitize_role(None) is None
    assert len(cg.sanitize_role("a" * 500)) <= cg.MAX_ROLE_CHARS


def test_run_node(monkeypatch):
    monkeypatch.setattr(cg, "_run_actor", lambda token, inp: MOCK_ITEMS)
    state = {"resume_analysis": RESUME, "job_match": MATCH, "skill_gaps": GAPS, "target_role": "Data Analyst"}
    out = cg.run(state)
    assert out["career_guidance"]["status"] == "ok"


# ---- HTTP layer (urlopen mocked) ----

class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_request_format(monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data),
                    method=req.get_method(), timeout=timeout)
        return FakeResp(json.dumps(MOCK_ITEMS).encode())

    monkeypatch.setattr(cg.urllib.request, "urlopen", fake)
    items = cg._run_actor("tok123", {"position": "x"})
    assert items == MOCK_ITEMS
    assert seen["url"].startswith("https://api.apify.com/v2/actors/misceres~indeed-scraper/run-sync-get-dataset-items?")
    assert "tok123" not in seen["url"]
    assert "maxTotalChargeUsd=0.25" in seen["url"] and "maxItems=10" in seen["url"]
    assert seen["method"] == "POST" and seen["body"] == {"position": "x"}
    assert seen["headers"]["Authorization"] == "Bearer tok123"


@pytest.mark.parametrize("http,code", [(429, "rate_limited"), (401, "auth_error"), (403, "auth_error"),
                                       (402, "quota_exceeded"), (404, "actor_unavailable"), (408, "timeout"),
                                       (400, "actor_error"), (500, "api_error")])
def test_http_status_mapping(monkeypatch, http, code):
    def boom(req, timeout):
        raise urllib.error.HTTPError("u", http, "x", {}, None)

    monkeypatch.setattr(cg.urllib.request, "urlopen", boom)
    with pytest.raises(cg.ApifyError) as e:
        cg._run_actor("tok123", {})
    assert e.value.code == code and "tok123" not in e.value.message


@pytest.mark.parametrize("exc,code", [(socket.timeout(), "timeout"),
                                      (urllib.error.URLError(socket.timeout()), "timeout"),
                                      (urllib.error.URLError("down"), "api_error"),
                                      (ConnectionResetError(), "api_error")])
def test_network_failures(monkeypatch, exc, code):
    def boom(req, timeout):
        raise exc

    monkeypatch.setattr(cg.urllib.request, "urlopen", boom)
    with pytest.raises(cg.ApifyError) as e:
        cg._run_actor("t", {})
    assert e.value.code == code


@pytest.mark.parametrize("body", [b"not json", b"{}", b'["x"]'])
def test_malformed_responses(monkeypatch, body):
    monkeypatch.setattr(cg.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    with pytest.raises(cg.ApifyError) as e:
        cg._run_actor("t", {})
    assert e.value.code == "invalid_response"