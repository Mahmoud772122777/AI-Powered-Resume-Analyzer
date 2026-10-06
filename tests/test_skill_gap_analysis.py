import io
import json
import urllib.error

import pytest

from agents import skill_gap_analysis as sg

RESUME = {
    "name": "Jane Secret",
    "contact": {"email": "jane@secret.com"},
    "skills": ["Python", "JS", "React", "Postgres"],
}
JD = "We need Python, JavaScript, React Native, PostgreSQL, Docker and Kubernetes."


JD_SKILLS = ["Python", "JavaScript", "React", "React Native", "PostgreSQL", "Docker", "Kubernetes"]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test-token-123")
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(sg, "load_dotenv", lambda *a, **k: None)


def make_embedder(vectors=None, error=None):
    calls = []

    def embedder(texts, token, model):
        calls.append(texts)
        if error:
            raise error
        table = vectors or {}
        return [table.get(t, [0.0, 1.0]) for t in texts]

    embedder.calls = calls
    return embedder


def test_deterministic_matching_and_semantic():
    emb = make_embedder({"kubernetes": [1.0, 0.0], "docker": [0.0, 1.0], "python": [0.0, 1.0]})
    out = sg.analyze_skill_gaps(RESUME, required_skills=JD_SKILLS, embedder=emb)
    assert out["status"] == "ok"
    d = out["data"]
    matched = {m["skill"] for m in d["matched"]}
    assert matched == {"python", "javascript", "react", "postgresql"}  # aliases normalized
    partial = {p["skill"]: p for p in d["partial"]}
    assert partial["react native"]["match_type"] == "related"
    assert partial["react native"]["resume_skill"] == "react"
    # docker is orthogonal-free: identical vector to "python" -> semantic partial
    assert partial["docker"]["match_type"] == "semantic"
    assert d["missing"] == ["kubernetes"]
    recs = {r["skill"]: r["priority"] for r in d["recommended_to_learn"]}
    assert recs == {"kubernetes": "high", "react native": "medium", "docker": "medium"}
    assert d["coverage_percent"] == round(100 * (4 + 0.5 * 2) / 7)
    assert len(emb.calls) == 1
    json.dumps(out)


def test_only_skills_are_sent_to_api():
    emb = make_embedder()
    sg.analyze_skill_gaps(RESUME, job_description=JD, embedder=emb)
    sent = " ".join(emb.calls[0])
    assert "Jane" not in sent and "secret" not in sent.lower()
    assert len(emb.calls[0]) <= sg.MAX_REQUIRED + sg.MAX_RESUME_SKILLS


def test_no_api_call_when_nothing_unmatched():
    emb = make_embedder()
    out = sg.analyze_skill_gaps(RESUME, required_skills=["python", "JS"], embedder=emb)
    assert out["data"]["missing"] == [] and emb.calls == []
    assert out["data"]["semantic"]["status"] == "skipped"


def test_resume_without_skills_makes_no_call_and_all_missing():
    emb = make_embedder()
    out = sg.analyze_skill_gaps({"name": "x"}, required_skills=["Python"], embedder=emb)
    assert out["data"]["missing"] == ["python"] and emb.calls == []


def test_missing_token_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    emb = make_embedder()
    out = sg.analyze_skill_gaps(RESUME, job_description=JD, embedder=emb)
    assert out["status"] == "partial" and out["error_code"] == "missing_api_token"
    assert out["data"]["missing"] == ["docker", "kubernetes"]
    assert emb.calls == []


@pytest.mark.parametrize("code", ["rate_limited", "auth_error", "model_unavailable", "invalid_response", "api_error"])
def test_api_errors_return_partial_with_deterministic_results(code):
    emb = make_embedder(error=sg.SkillApiError(code, "msg"))
    out = sg.analyze_skill_gaps(RESUME, job_description=JD, embedder=emb)
    assert out["status"] == "partial" and out["error_code"] == code
    assert out["data"]["semantic"]["status"] == "unavailable"
    assert {m["skill"] for m in out["data"]["matched"]} == {"python", "javascript", "react", "postgresql"}


def test_unexpected_exception_does_not_leak_token():
    out = sg.analyze_skill_gaps(RESUME, job_description=JD, embedder=make_embedder(error=RuntimeError("boom hf_test-token-123")))
    assert out["error_code"] == "api_error"
    assert "hf_test-token-123" not in json.dumps(out)


@pytest.mark.parametrize("kwargs,code", [
    ({"resume": None, "job_description": JD}, "invalid_resume"),
    ({"resume": {}, "job_description": JD}, "invalid_resume"),
    ({"resume": {"skills": "python"}, "job_description": JD}, "invalid_resume"),
    ({"resume": RESUME}, "empty_input"),
    ({"resume": RESUME, "job_description": "   "}, "empty_input"),
    ({"resume": RESUME, "job_description": "Looking for a friendly person"}, "no_required_skills"),
])
def test_invalid_inputs(kwargs, code):
    emb = make_embedder()
    out = sg.analyze_skill_gaps(embedder=emb, **kwargs)
    assert out["status"] == "error" and out["error_code"] == code
    assert emb.calls == []


def test_falls_back_to_job_match_required_skills():
    jm = {"data": {"matches": [{"score_breakdown": {"skills": {
        "basis": "job_required_skills", "matched": ["python"], "missing": ["rust"]}}}]}}
    out = sg.analyze_skill_gaps(RESUME, job_description="no known skills here", job_match=jm, embedder=make_embedder())
    assert out["data"]["required_skills_source"] == "job_match"
    assert out["data"]["required_skills"] == ["python", "rust"]


def test_skill_extraction_avoids_substring_false_positives():
    assert sg.extract_required_skills("Experience with JavaScript and MySQL") == ["javascript", "mysql"]


def test_input_bounds():
    many = [f"skill{i}" for i in range(200)]
    out = sg.analyze_skill_gaps({"skills": many}, required_skills=many, embedder=make_embedder())
    assert len(out["data"]["required_skills"]) == sg.MAX_REQUIRED
    assert len(out["data"]["resume_skills"]) == sg.MAX_RESUME_SKILLS


# ---- HTTP layer (urlopen mocked) ----

class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_request_format(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data), method=req.get_method())
        return FakeResp(json.dumps([[0.1, 0.2], [0.3, 0.4]]).encode())

    monkeypatch.setattr(sg.urllib.request, "urlopen", fake_urlopen)
    vecs = sg._embed_hf(["a", "b"], "hf_test-token-123", "BAAI/bge-small-en-v1.5")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert seen["url"] == "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
    assert seen["method"] == "POST" and seen["body"] == {"inputs": ["a", "b"]}
    assert seen["headers"]["Authorization"] == "Bearer hf_test-token-123"


@pytest.mark.parametrize("http,code", [(429, "rate_limited"), (401, "auth_error"), (403, "auth_error"),
                                       (404, "model_unavailable"), (503, "model_unavailable"), (500, "api_error")])
def test_http_status_mapping(monkeypatch, http, code):
    def boom(req, timeout):
        raise urllib.error.HTTPError("u", http, "x", {}, None)

    monkeypatch.setattr(sg.urllib.request, "urlopen", boom)
    with pytest.raises(sg.SkillApiError) as e:
        sg._embed_hf(["a"], "hf_test-token-123", "m")
    assert e.value.code == code and "hf_test-token-123" not in e.value.message


def test_network_failure(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(sg.urllib.request, "urlopen", boom)
    with pytest.raises(sg.SkillApiError) as e:
        sg._embed_hf(["a"], "t", "m")
    assert e.value.code == "api_error"


@pytest.mark.parametrize("body,n", [(b"not json", 1), (b"{}", 1), (b"[[0.1]]", 2), (b'[["x"]]', 1),
                                    (b"[[0.1],[0.2,0.3]]", 2), (b"[[]]", 1)])
def test_malformed_responses(monkeypatch, body, n):
    monkeypatch.setattr(sg.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    with pytest.raises(sg.SkillApiError) as e:
        sg._embed_hf(["a"] * n, "t", "m")
    assert e.value.code == "invalid_response"

def test_run_node(monkeypatch):
    monkeypatch.setattr(sg, "_embed_hf", lambda texts, token, model: [[0.0, 1.0] for _ in texts])
    state = {"resume_analysis": {"status": "ok", "data": RESUME}, "job_description": JD}
    out = sg.run(state)
    assert out["skill_gaps"]["status"] == "ok"


def test_run_node_without_valid_resume():
    out = sg.run({"resume_analysis": {"status": "error"}, "job_description": JD})
    assert out["skill_gaps"]["error_code"] == "invalid_resume"