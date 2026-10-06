"""Streamlit entry point for the AI-Powered Resume Analyzer."""
import streamlit as st

from orchestration import run_pipeline

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("AI-Powered Resume Analyzer and Job Matcher")
st.caption(
    "Resume analysis, job matching, skill-gap analysis, and career guidance "
    "with a bounded, traceable revision when job search can be broadened."
)

with st.form("resume_analysis"):
    resume_text = st.text_area("Paste your resume", height=200)
    job_description = st.text_area("Paste the job description", height=200)
    target_role = st.text_input("Target role (optional)")
    country = st.text_input("Country code for market data (optional)", max_chars=2)
    required_skills_text = st.text_input(
        "Required skills (optional; comma-separated)",
    )
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not resume_text.strip() or not job_description.strip():
        st.warning("Please provide both a resume and a job description.")
    else:
        required_skills = [
            skill.strip() for skill in required_skills_text.split(",") if skill.strip()
        ] or None
        result = run_pipeline(
            resume_text,
            job_description,
            target_role=target_role,
            country=country,
            required_skills=required_skills,
        )

        if result is None:
            st.error("The workflow did not return a result.")
        else:
            final_outcome = result.get("final_outcome") or {}
            statuses = final_outcome.get("agent_statuses") or {}
            st.subheader("Agent statuses")
            status_columns = st.columns(4)
            agent_labels = (
                ("resume_analysis", "Resume Analysis"),
                ("job_matching", "Job Matching"),
                ("skill_gap_analysis", "Skill Gap Analysis"),
                ("career_guidance", "Career Guidance"),
            )
            for column, (agent, label) in zip(status_columns, agent_labels):
                column.metric(label, statuses.get(agent, "unknown"))

            st.caption(
                f"Workflow outcome: {final_outcome.get('outcome', 'unknown')} · "
                f"Iterations: {result.get('iteration', 1)}"
            )

            st.subheader("Agent results")
            for key, label in (
                ("resume_analysis", "Resume Analysis"),
                ("job_match", "Job Matching"),
                ("skill_gaps", "Skill Gap Analysis"),
                ("career_guidance", "Career Guidance"),
            ):
                agent_result = result.get(key) or {}
                with st.expander(
                    f"{label} — {agent_result.get('status', 'not_run')}",
                    expanded=key == "career_guidance",
                ):
                    if agent_result.get("error"):
                        st.error(agent_result["error"])
                    if agent_result.get("warning"):
                        st.warning(agent_result["warning"])
                    if "data" in agent_result:
                        st.json(agent_result["data"])

            guidance = result.get("career_guidance") or {}
            guidance_data = guidance.get("data") or {}
            market = guidance_data.get("market") or {}
            st.subheader("Job-market data")
            if market.get("source") == "apify_live":
                st.success("Market data source: live Apify listings.")
            else:
                st.info("Market data source: fallback or unavailable; no live listings were returned.")

            st.subheader("Deterministic critique")
            critique = result.get("critique")
            if critique is not None:
                st.json(critique)
            else:
                st.info("Critique was not run because resume analysis did not succeed.")

            st.subheader("Revision history")
            revision_history = result.get("revision_history") or []
            if revision_history:
                st.json(revision_history)
            else:
                st.info("No revision was requested.")

            with st.expander("Execution trace"):
                st.json(result.get("trace") or [])
