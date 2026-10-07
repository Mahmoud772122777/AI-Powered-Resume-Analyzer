"""Streamlit entry point for the AI-Powered Resume Analyzer."""
import io
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from orchestration import run_pipeline

AGENTS = (
    ("resume_analysis", "resume_analysis", "Resume Analysis"),
    ("job_matching", "job_match", "Job Matching"),
    ("skill_gap_analysis", "skill_gaps", "Skill Gap Analysis"),
    ("career_guidance", "career_guidance", "Career Guidance"),
)

STATUS_DISPLAY = {
    "ok": ("✅", "Completed"),
    "partial": ("⚠️", "Partial"),
    "error": ("❌", "Failed"),
}
SKIPPED = ("⏭️", "Skipped")

FRIENDLY_ERRORS = {
    "missing_api_key": "This step is not set up yet because a required service key is missing. Ask the app owner to finish the configuration.",
    "missing_api_token": "This step is not set up yet because a required service token is missing. Ask the app owner to finish the configuration.",
    "auth_error": "The service rejected the app's credentials. Ask the app owner to check the configuration.",
    "rate_limited": "The service is receiving too many requests right now. Please wait a few minutes and try again.",
    "quota_exceeded": "The service's usage limit has been reached. Please try again later.",
    "server_error": "The service is temporarily unavailable. Please try again in a minute.",
    "api_error": "The service could not be reached or the request failed. Please try again.",
    "invalid_response": "The service sent back an answer the app could not read. Please try again.",
    "model_not_found": "The AI model configured for this step is not available. Ask the app owner to check the configuration.",
    "invalid_resume": "This step needs a successfully analyzed resume first.",
    "empty_input": "This step needs more input than was provided.",
    "no_input": "There was not enough output from earlier steps to continue.",
}
GENERIC_ERROR = "This step could not be completed. Please try again."


def status_display(raw_status: Any) -> tuple:
    return STATUS_DISPLAY.get(raw_status, SKIPPED)


def friendly_message(agent_result: Dict[str, Any]) -> Optional[str]:
    """User-facing text for a failed or partial agent result; None when there is no problem."""
    if not (agent_result.get("error") or agent_result.get("warning")):
        return None
    return FRIENDLY_ERRORS.get(agent_result.get("error_code"), GENERIC_ERROR)


def text_of(value: Any) -> str:
    if isinstance(value, dict):
        return " · ".join(str(v) for v in value.values() if v)
    return str(value) if value else ""


def bullet_list(items: List[Any]) -> None:
    for item in items:
        line = text_of(item)
        if line:
            st.markdown(f"- {line}")


def skill_names(items: Any) -> List[str]:
    names = []
    for item in items or []:
        name = item.get("skill") if isinstance(item, dict) else item
        if name:
            names.append(str(name))
    return names


def badge_text(value: Any) -> str:
    return str(value).replace("[", "(").replace("]", ")").replace(":", "-")


def chips(names: List[str], color: str = "blue") -> None:
    if names:
        st.markdown(" ".join(f":{color}-badge[{badge_text(n)}]" for n in names))


def pct(value: Any) -> str:
    return f"{value}%" if isinstance(value, (int, float)) else "n/a"


def period(item: Dict[str, Any]) -> str:
    return " – ".join(str(x) for x in (item.get("start_date"), item.get("end_date")) if x)


def render_problem(agent_result: Dict[str, Any]) -> None:
    message = friendly_message(agent_result)
    if not message:
        return
    if agent_result.get("status") == "error":
        st.error(message)
    else:
        st.warning(message + " Results below may be limited or based on fallback data.")


def render_resume(agent_result: Dict[str, Any]) -> None:
    render_problem(agent_result)
    data = agent_result.get("data")
    if not data:
        st.info("No resume details are available.")
        return
    st.subheader(data.get("name") or "Candidate")
    if data.get("professional_summary"):
        with st.container(border=True):
            st.markdown("**Professional summary**")
            st.write(data["professional_summary"])

    left, right = st.columns([2, 1], gap="large")
    with right:
        contact = {k: v for k, v in (data.get("contact") or {}).items() if v}
        if contact:
            with st.container(border=True):
                st.markdown("**Contact**")
                for key, value in contact.items():
                    st.markdown(f"{key.title()}: {value}")
    with left:
        skills = [str(s) for s in data.get("skills") or [] if s]
        st.markdown("##### Skills")
        if skills:
            chips(skills)
        else:
            st.caption("No skills were found in the resume.")

    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown("##### Education")
        education = data.get("education") or []
        if not education:
            st.caption("None found in the resume.")
        for item in education:
            with st.container(border=True):
                if isinstance(item, dict):
                    st.markdown(f"**{item.get('degree') or item.get('institution') or 'Education'}**"
                                + (f" · {item['field']}" if item.get("field") else ""))
                    sub = " · ".join(x for x in (item.get("institution") if item.get("degree") else None, period(item)) if x)
                    if sub:
                        st.caption(sub)
                else:
                    st.write(text_of(item))
        st.markdown("##### Certifications")
        certs = data.get("certifications") or []
        if not certs:
            st.caption("None found in the resume.")
        for item in certs:
            st.markdown(f"- {text_of(item)}")
    with cols[1]:
        st.markdown("##### Work experience")
        jobs = data.get("work_experience") or []
        if not jobs:
            st.caption("None found in the resume.")
        for item in jobs:
            with st.container(border=True):
                if isinstance(item, dict):
                    st.markdown(f"**{item.get('title') or 'Role'}**" + (f" · {item['company']}" if item.get("company") else ""))
                    if period(item):
                        st.caption(period(item))
                    if item.get("description"):
                        st.write(item["description"])
                else:
                    st.write(text_of(item))
        st.markdown("##### Projects")
        projects = data.get("projects") or []
        if not projects:
            st.caption("None found in the resume.")
        for item in projects:
            with st.container(border=True):
                if isinstance(item, dict):
                    st.markdown(f"**{item.get('name') or 'Project'}**")
                    if item.get("description"):
                        st.write(item["description"])
                    tech = item.get("technologies")
                    if isinstance(tech, list):
                        chips([str(t) for t in tech if t], "gray")
                    elif tech:
                        st.caption(f"Technologies: {tech}")
                else:
                    st.write(text_of(item))


def render_jobs(agent_result: Dict[str, Any]) -> None:
    data = agent_result.get("data") or {}
    matches = data.get("matches") or []
    if agent_result.get("status") == "error" or (agent_result.get("error") and not matches):
        st.warning("Job matching is temporarily unavailable, so no job matches can be shown. "
                   + (friendly_message(agent_result) or "") + " Technical details are under Details.")
        return
    render_problem(agent_result)
    if data.get("query"):
        st.caption(f"Search used: {data['query']}")
    if not matches:
        st.info(data.get("message") or "No matching jobs were found for this search.")
        return
    st.markdown(f"##### {len(matches)} job match{'es' if len(matches) != 1 else ''}")
    for job in matches:
        score = job.get("match_score")
        breakdown = job.get("score_breakdown") or {}
        skills = breakdown.get("skills") or {}
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"#### {job.get('title') or 'Untitled position'}")
            left.caption(" · ".join(x for x in (job.get("company"), job.get("location"), job.get("employment_type")) if x))
            right.metric("Match score", pct(score))
            if isinstance(score, (int, float)):
                right.progress(max(0, min(100, int(score))) / 100)
            parts = [f"{name.title()}: {breakdown[name]['score']}%"
                     for name in ("skills", "experience", "qualifications")
                     if isinstance(breakdown.get(name), dict) and isinstance(breakdown[name].get("score"), (int, float))]
            if parts:
                st.caption("Score breakdown — " + " · ".join(parts))
            matched, missing = skill_names(skills.get("matched")), skill_names(skills.get("missing"))
            c1, c2 = st.columns(2)
            if matched:
                c1.markdown("**Matched skills**")
                with c1:
                    chips(matched, "green")
            if missing:
                c2.markdown("**Missing skills**")
                with c2:
                    chips(missing, "orange")
            if job.get("apply_link"):
                st.link_button("Apply", job["apply_link"])


def render_gaps(agent_result: Dict[str, Any]) -> None:
    render_problem(agent_result)
    data = agent_result.get("data")
    if not data:
        st.info("No skill gap results are available.")
        return
    coverage = data.get("coverage_percent")
    if isinstance(coverage, (int, float)):
        st.metric("Skill coverage", pct(coverage))
        st.progress(max(0, min(100, int(coverage))) / 100)
    matched, partial, missing = (skill_names(data.get(k)) for k in ("matched", "partial", "missing"))
    left, right = st.columns(2, gap="large")
    with left.container(border=True):
        st.markdown("##### Your strengths")
        if matched:
            chips(matched, "green")
        else:
            st.caption("No matched skills.")
    with right.container(border=True):
        st.markdown("##### Skills to improve")
        if partial:
            st.markdown("**Partially matched**")
            chips(partial, "yellow")
        if missing:
            st.markdown("**Missing**")
            chips(missing, "orange")
        if not partial and not missing:
            st.caption("No gaps found.")
    recommendations = data.get("recommended_to_learn") or []
    if recommendations:
        st.markdown("##### Recommended next skills")
        for rec in recommendations:
            with st.container(border=True):
                if isinstance(rec, dict):
                    st.markdown(f"**{rec.get('skill') or 'Skill'}**" + (f" · {rec['priority']} priority" if rec.get("priority") else ""))
                    if rec.get("reason"):
                        st.caption(rec["reason"])
                else:
                    st.markdown(f"**{rec}**")


def render_guidance(agent_result: Dict[str, Any]) -> None:
    render_problem(agent_result)
    data = agent_result.get("data")
    if not data:
        st.info("No career guidance is available.")
        return
    snapshot = data.get("profile_snapshot") or {}
    st.markdown("##### Profile snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target role", snapshot.get("target_role") or data.get("target_role") or "n/a")
    c2.metric("Best job match", pct(snapshot.get("best_match_score")))
    c3.metric("Skill coverage", pct(snapshot.get("skill_coverage_percent")))

    sections = (
        ("skills_to_prioritize", "Skills to prioritize"),
        ("resume_improvements", "Resume improvements"),
        ("project_ideas", "Project ideas"),
        ("job_search_suggestions", "Job search suggestions"),
    )
    available = [(k, label) for k, label in sections if data.get(k)]
    for i in range(0, len(available), 2):
        cols = st.columns(2, gap="large")
        for col, (key, label) in zip(cols, available[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"**{label}**")
                for item in data[key]:
                    if isinstance(item, dict):
                        detail = " — ".join(x for x in (item.get("skill"), item.get("reason")) if x)
                        st.markdown(f"- {detail or text_of(item)}")
                    else:
                        st.markdown(f"- {item}")

    market = data.get("market") or {}
    st.markdown("##### Market information")
    if market.get("source") == "apify_live":
        st.success("Live: based on current job listings.")
        st.write(f"{market.get('listing_count', 0)} live listings reviewed.")
        for entry in market.get("top_skills") or []:
            note = "on your resume" if entry.get("in_resume") else "not on your resume"
            st.markdown(f"- {entry.get('skill')}: {entry.get('percent')}% of listings ({note})")
        for listing in market.get("sample_listings") or []:
            line = " · ".join(x for x in (listing.get("title"), listing.get("company"), listing.get("location"), listing.get("salary")) if x)
            if listing.get("url"):
                st.markdown(f"- [{line or 'Listing'}]({listing['url']})")
            elif line:
                st.markdown(f"- {line}")
    elif data.get("guidance_source") == "fallback_no_market_data":
        st.warning("Fallback: live market data was not available, so this guidance uses only your resume, job match and skill gap results.")
    else:
        st.info("Unavailable: no market information was returned.")


def render_status_cards(result: Dict[str, Any], statuses: Dict[str, Any]) -> None:
    columns = st.columns(4)
    for column, (status_key, result_key, label) in zip(columns, AGENTS):
        raw = statuses.get(status_key) or (result.get(result_key) or {}).get("status")
        icon, text = status_display(raw)
        agent_result = result.get(result_key) or {}
        with column.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(f"### {icon} {text}")
            if raw == "partial":
                st.caption("Used fallback or limited data.")
            elif raw == "error":
                st.caption(friendly_message(agent_result) or GENERIC_ERROR)
            elif raw not in STATUS_DISPLAY:
                st.caption("Not run.")


def workflow_summary(result: Dict[str, Any], statuses: Dict[str, Any]) -> None:
    values = [statuses.get(s) or (result.get(r) or {}).get("status") for s, r, _ in AGENTS]
    if values and all(v == "ok" for v in values):
        st.success("Analysis completed successfully.")
    elif values and values[0] != "ok":
        st.error("Analysis could not be completed. Your resume could not be analyzed, so the remaining steps were skipped.")
    else:
        st.warning("Analysis completed with some limitations. Steps marked Partial or Failed are explained in each tab.")
    revisions = result.get("revision_history") or []
    if revisions:
        reasons = ", ".join(str(r.get("reason", "unknown")).replace("_", " ") for r in revisions if isinstance(r, dict))
        st.info(f"The job search was revised once ({reasons}).")
    outcome = (result.get("final_outcome") or {}).get("outcome")
    internal = ", ".join(f"{k}: {v}" for k, v in statuses.items())
    if outcome or internal:
        st.caption("Workflow outcome: " + (str(outcome) if outcome else "n/a") + (f" · Internal statuses — {internal}" if internal else ""))


def render_details(result: Dict[str, Any]) -> None:
    st.caption("Technical information for troubleshooting.")
    with st.expander("Critique"):
        critique = result.get("critique")
        if critique is not None:
            st.json(critique)
        else:
            st.info("Critique was not run because resume analysis did not succeed.")
    with st.expander("Revision history"):
        history = result.get("revision_history") or []
        if history:
            st.json(history)
        else:
            st.info("No revision was requested.")
    with st.expander("Technical errors and warnings"):
        found = False
        for _, result_key, label in AGENTS:
            agent_result = result.get(result_key) or {}
            for field in ("error_code", "error", "warning"):
                if agent_result.get(field):
                    found = True
                    st.write(f"**{label}** · {field}: {agent_result[field]}")
        if not found:
            st.caption("None.")
    with st.expander("Execution trace"):
        st.json(result.get("trace") or [])
    with st.expander("Raw JSON"):
        st.json({result_key: result.get(result_key) for _, result_key, _ in AGENTS})

st.set_page_config(
    page_title="AI-Powered Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide",
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ResumeFileError(Exception):
    """Raised with a user-friendly message when an uploaded resume cannot be read."""


def extract_resume_text(filename: str, data: bytes) -> str:
    """Extract text locally from an uploaded TXT or PDF resume."""
    name = (filename or "").lower()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ResumeFileError("The file is too large. Please upload a file under 5 MB or paste the text instead.")
    if name.endswith(".txt"):
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ResumeFileError("The text file could not be read. Please save it as UTF-8 or paste the text instead.")
    elif name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ResumeFileError("This PDF is password protected. Please remove the password or paste the text instead.")
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except ResumeFileError:
            raise
        except Exception:
            raise ResumeFileError("This PDF could not be read. It may be damaged. Please try another file or paste the text instead.")
    else:
        raise ResumeFileError("Unsupported file type. Please upload a PDF or TXT file.")
    text = text.strip()
    if not text:
        raise ResumeFileError("No text could be found in this file. If it is a scanned PDF, please paste the text instead.")
    return text


def resolve_resume_input(uploaded_name: Optional[str], uploaded_bytes: Optional[bytes], pasted_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (resume_text, error). Exactly one of the two is set."""
    pasted = (pasted_text or "").strip()
    if uploaded_name and pasted:
        return None, "You provided both an uploaded file and pasted text. Please use only one: remove the file or clear the text box."
    if uploaded_name:
        try:
            return extract_resume_text(uploaded_name, uploaded_bytes or b""), None
        except ResumeFileError as exc:
            return None, str(exc)
    if pasted:
        return pasted, None
    return None, None


FORM_KEYS = ("resume_text", "job_description", "target_role", "country", "required_skills_text")


def clear_form():
    for key in FORM_KEYS:
        st.session_state[key] = ""
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


st.markdown(
    """
    <style>
    button[data-testid="stBaseButton-primaryFormSubmit"],
    button[kind="primaryFormSubmit"] {
        background-color: #1f4e8c;
        border-color: #1f4e8c;
        color: #ffffff;
    }
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        background-color: #173b6b;
        border-color: #173b6b;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "Paste your resume and a job description, then press **Analyze My Resume**. "
        "The agents run in order and the results appear in tabs."
    )
    st.divider()
    st.subheader("4-Agent Architecture")
    st.markdown(
        "1. **Resume Analysis** extracts your skills, education and experience.\n"
        "2. **Job Matching** searches for jobs and scores each match.\n"
        "3. **Skill Gap Analysis** compares your skills with the required ones.\n"
        "4. **Career Guidance** suggests next steps and shows market information when available."
    )
    st.caption("A LangGraph workflow runs the agents and may re-run Job Matching and Skill Gap once if a recoverable issue is found.")
    st.divider()
    st.subheader("🔒 Privacy")
    st.markdown(
        "Your resume text is sent to the AI service that analyzes it. "
        "Job search and market-data services receive only a role, keywords or a country code, "
        "not your resume or personal details."
    )
    st.divider()
    st.subheader("⚙️ Configuration")
    st.markdown(
        "The app owner provides the service keys in a private `.env` file. "
        "If a service is unavailable, the app shows partial results where it can."
    )

st.title("AI-Powered Resume Analyzer & Job Matcher")
st.markdown("##### From resume to opportunities: analyze, match, close skill gaps and plan your next step.")
st.caption("4 AI Agents • LangGraph • Resume → Jobs → Skills → Career")
st.divider()

with st.form("resume_analysis", border=False):
    st.subheader("1. Your documents")
    left, right = st.columns(2, gap="large")
    with left.container(border=True):
        st.markdown("**📄 Resume**")
        st.caption("Upload your resume or paste the text below. Use one method, not both.")
        uploaded_file = st.file_uploader(
            "Upload a resume (PDF or TXT)",
            type=["pdf", "txt"],
            key=f"resume_file_{st.session_state.get('upload_version', 0)}",
            help="The file is read on this device and only the extracted text is analyzed. The file itself is not sent anywhere.",
        )
        resume_text = st.text_area(
            "Or paste your resume text",
            key="resume_text",
            height=280,
            placeholder=(
                "Example:\n"
                "Jane Doe | jane@example.com\n"
                "Skills: Python, SQL, Pandas\n"
                "Experience: Data Analyst, ABC Corp (2022-2024)\n"
                "Education: B.Sc. Computer Science"
            ),
            help="Paste the full text of your resume. Plain text works best.",
        )
    with right.container(border=True):
        job_description = st.text_area(
            "💼 Job description",
            key="job_description",
            height=280,
            placeholder=(
                "Example:\n"
                "Data Analyst\n"
                "We are looking for an analyst with Python, SQL and "
                "dashboard experience to join our analytics team."
            ),
            help="Paste the job posting you are interested in. It is used to find similar jobs and compare skills.",
        )

    st.subheader("2. Optional details")
    st.caption("All fields below are optional. Leave them blank to use sensible defaults.")
    with st.container(border=True):
        o1, o2, o3 = st.columns([2, 1, 3], gap="medium")
        target_role = o1.text_input(
            "Target role",
            key="target_role",
            placeholder="e.g. Data Analyst",
            help="Leave blank to use the first line of the job description.",
        )
        country = o2.text_input(
            "Country code",
            key="country",
            max_chars=2,
            placeholder="e.g. IN",
            help="Two-letter country code, used for market data.",
        )
        required_skills_text = o3.text_input(
            "Required skills",
            key="required_skills_text",
            placeholder="e.g. Python, SQL, Excel",
            help="Comma-separated. Leave blank to derive them from the job description.",
        )

    b1, b2 = st.columns([4, 1])
    submitted = b1.form_submit_button("Analyze My Resume", type="primary", use_container_width=True)
    b2.form_submit_button("Clear", on_click=clear_form, use_container_width=True)
if submitted:
    resume_for_analysis, resume_error = resolve_resume_input(
        uploaded_file.name if uploaded_file else None,
        uploaded_file.getvalue() if uploaded_file else None,
        resume_text,
    )
    if resume_error:
        st.warning(resume_error)
    elif not resume_for_analysis or not job_description.strip():
        st.warning("Please provide both a resume and a job description.")
    else:
        if uploaded_file:
            st.caption(f"Resume loaded: {uploaded_file.name} ({len(resume_for_analysis):,} characters extracted)")
        required_skills = [
            skill.strip() for skill in required_skills_text.split(",") if skill.strip()
        ] or None
        with st.spinner("Analyzing your resume. This can take a minute..."):
            result = run_pipeline(
                resume_for_analysis,
                job_description,
                target_role=target_role,
                country=country,
                required_skills=required_skills,
            )

        if result is None:
            st.error("The analysis did not return a result. Please try again.")
        else:
            final_outcome = result.get("final_outcome") or {}
            statuses = final_outcome.get("agent_statuses") or {}

            st.header("Analysis Overview")
            workflow_summary(result, statuses)
            render_status_cards(result, statuses)

            tabs = st.tabs(["Resume Analysis", "Job Matches", "Skill Gap", "Career Guidance", "Details"])
            with tabs[0]:
                render_resume(result.get("resume_analysis") or {})
            with tabs[1]:
                render_jobs(result.get("job_match") or {})
            with tabs[2]:
                render_gaps(result.get("skill_gaps") or {})
            with tabs[3]:
                render_guidance(result.get("career_guidance") or {})
            with tabs[4]:
                render_details(result)
