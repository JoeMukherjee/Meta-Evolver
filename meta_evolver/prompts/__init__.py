"""System-prompt scaffolding and OPRO-style prompt evolution."""
from meta_evolver.prompts.optimizer import PromptCandidate, PromptOptimizer
from meta_evolver.prompts.templates import (
    BASE_SYSTEM_PROMPT,
    ensure_placeholders,
    render_system_prompt,
)

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "PromptCandidate",
    "PromptOptimizer",
    "ensure_placeholders",
    "render_system_prompt",
]
