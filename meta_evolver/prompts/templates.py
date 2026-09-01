"""Prompt scaffolding.

The base system prompt is deliberately domain-neutral. Anything specific to a
benchmark arrives from the benchmark adapter, and anything learned arrives from
the memory bank or the prompt optimizer. That separation is what lets one
engine drive an SRE incident simulator and a text adventure without a branch.

``{memory_section}`` and ``{guidance_section}`` are the two runtime injection
points. The optimizer must preserve them; ``ensure_placeholders`` enforces
that, because an evolved prompt that quietly drops them disables retrieval and
exploration guidance while still scoring as "an improvement".
"""
from __future__ import annotations

MEMORY_PLACEHOLDER = "{memory_section}"
GUIDANCE_PLACEHOLDER = "{guidance_section}"

BASE_SYSTEM_PROMPT = """\
You are an autonomous problem-solving agent operating inside an instrumented
environment. You act only through the tools you have been given, one call per
turn, and you learn what is true from the observations they return.

Method:
1. Read the current observation before acting. State briefly what it rules in
   or out. Do not restate the plan you already had.
2. Form one hypothesis and choose the single action that would most cheaply
   confirm or refute it.
3. Treat an error or a timeout as missing information, not as a fact. Retry it
   or reach the same fact another way before concluding anything from it.
4. Change the world only after you understand it, and verify every change took
   effect before you rely on it.
5. Never repeat an action that produced no new information. If two consecutive
   actions taught you nothing, your hypothesis is wrong -- change it.
6. Finish only when the environment's own check confirms the goal, not when
   the fix looks right to you.

{memory_section}
{guidance_section}"""


def ensure_placeholders(prompt: str) -> str:
    """Guarantee both injection points survive an optimization round."""
    out = prompt
    if MEMORY_PLACEHOLDER not in out:
        out = out.rstrip() + "\n\n" + MEMORY_PLACEHOLDER
    if GUIDANCE_PLACEHOLDER not in out:
        out = out.rstrip() + "\n" + GUIDANCE_PLACEHOLDER
    return out


def render_system_prompt(
    template: str, memory_section: str = "", guidance_section: str = ""
) -> str:
    """Fill the injection points.

    Uses targeted replacement rather than ``str.format`` because evolved
    prompts routinely contain literal braces -- JSON examples, code snippets --
    and ``format`` would raise ``KeyError`` on them, discarding the prompt for
    a reason unrelated to its quality.
    """
    filled = ensure_placeholders(template)
    filled = filled.replace(MEMORY_PLACEHOLDER, memory_section or "")
    filled = filled.replace(GUIDANCE_PLACEHOLDER, guidance_section or "")
    return "\n".join(line.rstrip() for line in filled.splitlines()).strip() + "\n"
