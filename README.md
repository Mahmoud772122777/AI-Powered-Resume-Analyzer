# AI-Powered-Resume-Analyzer
AI-powered resume analysis, job matching, skill gap detection, and career guidance using four collaborating AI agents.

## Project Structure

```
app.py              Streamlit UI
agents/             Four agents (resume analysis, job matching, skill gap, career guidance)
orchestration/      LangGraph workflow wiring the agents
tests/              pytest tests
```

## Setup (Windows, Python 3.10)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # replace placeholders with your own provider credentials
```

The workflow uses `GEMINI_API_KEY` for resume analysis, `JSEARCH_API_KEY` for job
matching, `HF_TOKEN` for semantic skill comparison, and `APIFY_API_TOKEN` for
career-guidance market data. Missing or unavailable market data results in
fallback guidance. `GEMINI_MODEL` and `HF_MODEL` are optional overrides.

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
pytest
```

## Workflow behavior

The four agents retain separate responsibilities: Gemini parses the resume,
JSearch returns job listings that are scored locally against the parsed resume,
skill-gap analysis compares required and resume skills (using Hugging Face only
for unresolved skill names when configured), and Apify provides optional market
listings for career guidance. JSearch receives a role or a job-description query,
not resume text or personal information. Apify receives a sanitized role and
optional country code, not resume text or personal information.

A deterministic orchestration critique can request at most one revision, and
only when an empty job search can be retried using a different query derived
from the supplied job description. The revision reruns Job Matching and Skill
Gap Analysis, never Resume Analysis. The UI displays agent statuses and results,
the critique, revision history, market-data source, and execution trace.

Tests mock external provider calls; successful tests do not verify live API
credentials, provider access, or real market-data availability.