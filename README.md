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
copy .env.example .env   # then add your own GEMINI_API_KEY
```

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
pytest
```

## Status

Initial scaffold only: agents return placeholder data and the Gemini API is not called yet.