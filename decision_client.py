"""
Backward-compatibility facade for System 1 Ollama decision client.
All LLM client logic has been modularized under `llm.decision_client`.
"""

from llm.decision_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DecisionResult,
    LocalFastSimulator,
    SystemOneAgent,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DecisionResult",
    "LocalFastSimulator",
    "SystemOneAgent",
]
