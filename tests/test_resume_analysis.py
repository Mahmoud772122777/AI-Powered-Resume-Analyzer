import json

import pytest

from agents import resume_analysis as ra


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setattr(ra, "load_dotenv", lambda *a, **k: None)


def _caller(response):
    calls = []

    def caller(prompt, key, model):
        calls.append(prompt)
        if isinstance(response, Exception):
            raise response
        return response

    caller.calls = calls
    return caller


def test_success_extracts_and_normalizes():
    payload = {
        "name": "Jane Doe",
        "contact": {"email": "jane@x.com"},
        "skills": ["Python", " SQL ", ""],
        "education": [{"institution": "MIT"}],
        "unexpected": "ignored",
    }
    c = _caller("```json\n" + json.dumps(payload) + "\n```")
    out = ra.analyze_resume("Jane Doe resume", caller=c)
    assert out["status"] == "ok"
    d = out["data"]
    assert d["name"] == "Jane Doe"
    assert d["contact"]["email"] == "jane@x.com"
    assert d["contact"]["phone"] is None
    assert d["skills"] == ["Python", "SQL"]
    assert d["projects"] == [] and d["certifications"] == []
    assert d["professional_summary"] is None
    assert "unexpected" not in d
    json.dumps(out)
    assert len(c.calls) == 1


def test_default_model_is_gemini_38_flash(monkeypatch):
    seen = {}
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(
        ra, "_call_gemini",
        lambda prompt, key, model: (seen.update(model=model) or "{}"),
    )

    out = ra.analyze_resume("resume")

    assert out["status"] == "ok"
    assert seen["model"] == "gemini-3.8-flash"


def test_model_is_loaded_from_project_dotenv(monkeypatch):
    seen = {}

    def load_project_env(dotenv_path):
        seen["dotenv_path"] = dotenv_path
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    monkeypatch.setattr(ra, "load_dotenv", load_project_env)
    monkeypatch.setattr(
        ra, "_call_gemini",
        lambda prompt, key, model: (seen.update(model=model) or "{}"),
    )

    out = ra.analyze_resume("resume")

    assert out["status"] == "ok"
    assert seen["dotenv_path"] == ra.ENV_FILE
    assert seen["model"] == "gemini-3.8-flash"


def test_empty_model_setting_uses_default(monkeypatch):
    seen = {}
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    monkeypatch.setattr(
        ra, "_call_gemini",
        lambda prompt, key, model: (seen.update(model=model) or "{}"),
    )

    out = ra.analyze_resume("resume")

    assert out["status"] == "ok"
    assert seen["model"] == ra.DEFAULT_MODEL


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_empty_input_makes_no_api_call(text):
    c = _caller("{}")
    out = ra.analyze_resume(text, caller=c)
    assert out["error_code"] == "empty_input"
    assert c.calls == []


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    c = _caller("{}")
    out = ra.analyze_resume("resume", caller=c)
    assert out["error_code"] == "missing_api_key"
    assert c.calls == []


def test_invalid_response():
    for bad in ("not json", "[1, 2]", ""):
        assert ra.analyze_resume("resume", caller=_caller(bad))["error_code"] == "invalid_response"


def test_quota_error():
    c = _caller(Exception("429 quota exceeded"))
    out = ra.analyze_resume("resume", caller=c)
    assert out["error_code"] == "quota_exceeded"
    assert len(c.calls) == 1


def test_api_error_does_not_leak_secret():
    out = ra.analyze_resume("resume", caller=_caller(RuntimeError("boom test-key-123")))
    assert out["error_code"] == "api_error"
    assert "test-key-123" not in json.dumps(out)


def test_model_not_found():
    class FakeApiError(Exception):
        code = 404

    out = ra.analyze_resume("resume", caller=_caller(FakeApiError("404 NOT_FOUND test-key-123")))
    assert out["error_code"] == "model_unavailable"
    assert "GEMINI_MODEL" in out["error"]
    assert "test-key-123" not in json.dumps(out)


def test_quota_error_by_status_code():
    class FakeApiError(Exception):
        code = 429

    out = ra.analyze_resume("resume", caller=_caller(FakeApiError("slow down")))
    assert out["error_code"] == "quota_exceeded"


def test_gemini_server_error_is_reported_without_provider_details(monkeypatch):
    class FakeServerError(Exception):
        code = 500

    monkeypatch.setattr(ra.time, "sleep", lambda _delay: None)
    out = ra.analyze_resume("resume", caller=_caller(FakeServerError("internal details")))

    assert out["error_code"] == "server_error"
    assert "internal details" not in json.dumps(out)


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_temporary_server_error_retries_then_succeeds(monkeypatch, status_code):
    class FakeServerError(Exception):
        code = status_code

    calls = []
    delays = []

    def caller(prompt, key, model):
        calls.append((prompt, key, model))
        if len(calls) == 1:
            raise FakeServerError("temporarily unavailable")
        return '{"name": "Jane"}'

    monkeypatch.setattr(ra.time, "sleep", delays.append)

    out = ra.analyze_resume("resume", caller=caller)

    assert out["status"] == "ok"
    assert out["data"]["name"] == "Jane"
    assert len(calls) == 2
    assert delays == [ra.SERVER_ERROR_RETRY_BACKOFF_SECONDS]


def test_temporary_server_error_stops_after_two_retries(monkeypatch):
    class FakeServerError(Exception):
        code = 504

    calls = []
    delays = []

    def caller(prompt, key, model):
        calls.append((prompt, key, model))
        raise FakeServerError("temporarily unavailable")

    monkeypatch.setattr(ra.time, "sleep", delays.append)

    out = ra.analyze_resume("resume", caller=caller)

    assert out["error_code"] == "server_error"
    assert len(calls) == ra.MAX_SERVER_ERROR_RETRIES + 1
    assert delays == [
        ra.SERVER_ERROR_RETRY_BACKOFF_SECONDS,
        ra.SERVER_ERROR_RETRY_BACKOFF_SECONDS * 2,
    ]


@pytest.mark.parametrize("status_code", [400, 401, 403, 429, 501, 505])
def test_permanent_or_auth_errors_are_not_retried(monkeypatch, status_code):
    class FakeApiError(Exception):
        code = status_code

    calls = []
    monkeypatch.setattr(ra.time, "sleep", lambda _delay: pytest.fail("must not sleep"))

    def caller(prompt, key, model):
        calls.append((prompt, key, model))
        raise FakeApiError("request failed")

    out = ra.analyze_resume("resume", caller=caller)

    assert len(calls) == 1
    if status_code == 429:
        assert out["error_code"] == "quota_exceeded"
    else:
        assert out["error_code"] == "api_error"


def test_gemini_call_uses_google_genai_sdk(monkeypatch):
    from google import genai

    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return type("R", (), {"text": "{}"})()

    class FakeClient:
        def __init__(self, api_key):
            seen["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert ra._call_gemini("p", "k", "m") == "{}"
    assert seen["model"] == "m" and seen["api_key"] == "k"
    assert seen["contents"] == "p"
    assert seen["config"] == {"temperature": 0.0}


def test_run_node_updates_state_key(monkeypatch):
    monkeypatch.setattr(ra, "_call_gemini", lambda p, k, m: json.dumps({"name": "A"}))
    out = ra.run({"resume_text": "text"})
    assert out["resume_analysis"]["data"]["name"] == "A"
def test_failure_diagnostics_log_status_without_secrets(monkeypatch, caplog):
    import logging

    class FakeServerError(Exception):
        code = 503
        status = "UNAVAILABLE"
        message = "model overloaded key=test-key-123 John Doe jane@example.com is the candidate"

    def boom(prompt, api_key, model_name):
        raise FakeServerError("x")

    monkeypatch.setattr(ra, "_call_gemini", boom)
    monkeypatch.setattr(ra.time, "sleep", lambda s: None)
    resume = "John Doe jane@example.com is the candidate\nPython developer with five years experience"
    with caplog.at_level(logging.WARNING, logger=ra.logger.name):
        result = ra.analyze_resume(resume)
    text = caplog.text
    assert result["error_code"] == "server_error"
    assert "http_status=503" in text and "UNAVAILABLE" in text
    assert "attempt=3/3" in text
    assert "test-key-123" not in text
    assert "jane@example.com" not in text
    assert "John Doe" not in text


def test_prompt_requests_json_only():
    assert "valid JSON only" in ra.PROMPT
    assert "no code fences" in ra.PROMPT


@pytest.mark.parametrize(
    "wrap",
    [
        lambda s: s,
        lambda s: "```json\n" + s + "\n```",
        lambda s: "```\n" + s + "\n```",
        lambda s: "Here is the result:\n```json\n" + s + "\n```\nDone.",
    ],
)
def test_plain_and_fenced_json_parse(wrap):
    c = _caller(wrap(json.dumps({"name": "Test User", "skills": ["Python"]})))
    out = ra.analyze_resume("resume", caller=c)
    assert out["status"] == "ok"
    assert out["data"]["skills"] == ["Python"]


@pytest.mark.parametrize("bad", ["{\"name\": ", "```json\n{bad json}\n```", "```json\n[1]\n```"])
def test_malformed_fenced_or_plain_json_is_invalid_response(bad):
    out = ra.analyze_resume("resume", caller=_caller(bad))
    assert out["error_code"] == "invalid_response"
