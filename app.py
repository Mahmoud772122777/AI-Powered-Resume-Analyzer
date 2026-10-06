"""Streamlit entry point for the AI-Powered Resume Analyzer."""
import streamlit as st

from orchestration import run_pipeline

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("AI-Powered Resume Analyzer and Job Matcher")
st.caption("Initial scaffold: agents are placeholders and no Gemini calls are made yet.")

resume_text = st.text_area("Paste your resume", height=200)
job_description = st.text_area("Paste the job description", height=200)

if st.button("Analyze"):
    if not resume_text.strip() or not job_description.strip():
        st.warning("Please provide both a resume and a job description.")
    else:
        st.json(run_pipeline(resume_text, job_description))