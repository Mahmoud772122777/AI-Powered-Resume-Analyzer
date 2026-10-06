"""LangGraph workflow connecting the four agents sequentially (placeholder)."""
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import (  # noqa: F401  (modules imported for node wiring)
    career_guidance,
    job_matching,
    resume_analysis,
    skill_gap_analysis,
)


class AnalyzerState(TypedDict, total=False):
    resume_text: str
    job_description: str
    resume_analysis: Dict[str, Any]
    job_match: Dict[str, Any]
    skill_gaps: Dict[str, Any]
    career_guidance: Dict[str, Any]


def build_graph():
    """Build and compile the workflow graph."""
    graph = StateGraph(AnalyzerState)
    graph.add_node("resume_analysis", resume_analysis.run)
    graph.add_node("job_matching", job_matching.run)
    graph.add_node("skill_gap_analysis", skill_gap_analysis.run)
    graph.add_node("career_guidance", career_guidance.run)

    graph.set_entry_point("resume_analysis")
    graph.add_edge("resume_analysis", "job_matching")
    graph.add_edge("job_matching", "skill_gap_analysis")
    graph.add_edge("skill_gap_analysis", "career_guidance")
    graph.add_edge("career_guidance", END)
    return graph.compile()


def run_pipeline(resume_text: str, job_description: str) -> Optional[Dict[str, Any]]:
    """Run the full pipeline and return the final state."""
    app = build_graph()
    return app.invoke(
        {"resume_text": resume_text, "job_description": job_description}
    )