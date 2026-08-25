"""Small, extensible IT skill alias taxonomy for deterministic normalization."""

from __future__ import annotations

SKILL_ALIASES: dict[str, str] = {
    "restful api": "REST API", "restful apis": "REST API", "rest apis": "REST API", "rest api": "REST API",
    "git/github": "Git", "git / github": "Git", "github": "Git", "git": "Git",
    "pho bert": "PhoBERT", "phobert": "PhoBERT",
    "llms": "LLM", "llm": "LLM", "cnn": "CNN", "cnns": "CNN",
    "auto gen": "AutoGen", "autogen": "AutoGen",
    "scikit learn": "Scikit-learn", "scikit-learn": "Scikit-learn",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
}

# Related concepts are intentionally not aliases and are not collapsed.
RELATED_CONCEPTS: dict[str, set[str]] = {
    "agentic/LLM frameworks": {"LangChain", "Google ADK", "AutoGen"},
}


def alias_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())
