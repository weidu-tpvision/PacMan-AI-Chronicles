"""
LLM / System 1 client module for Ollama fast inference and simulation.
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
