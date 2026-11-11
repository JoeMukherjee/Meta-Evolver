"""EvidenceReducer -- SoL-Pi's Evidence-Preserving Reducer, applied to large
tool observations before the main policy model reads them.

This project's existing default is flat truncation: ``act()`` cuts an
observation to ``observation_chars`` and keeps whichever prefix that happens
to land on. For something like ``devops.py``'s ``inspect_service_logs`` --
the concrete example the vault's SoL-Pi analysis names -- the line that
actually matters (the one error, the one out-of-range value) is exactly as
likely to be past the cutoff as before it. A reducer asks a cheap model to
keep *evidence* specifically -- errors, statuses, specific values -- and
compress everything else, rather than keeping a fixed prefix and discarding
the rest regardless of what it contains.

Deliberately narrow: this is a real new model call (SoL-Pi's own reported
numbers -- 49% fewer tokens, 33% lower cost overall -- are for the whole
observation this trades against being large in the first place, not free),
so it only fires above ``min_chars``, and it degrades to plain truncation on
any failure rather than ever costing the episode. Off by default, wired in
as an optional argument to ``build_episode_graph``, exactly like the other
Phase 3 mechanisms in this project.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from meta_evolver.llm.client import invoke_text

_REDUCER_SYSTEM = (
    "You compress a tool's raw output into a short, verified receipt for "
    "another agent that will act on it. Keep every concrete piece of "
    "evidence: error messages, status codes, specific numbers, names, and "
    "anything that looks like the actual answer to what was being checked. "
    "Drop repeated boilerplate, padding, and lines that carry no "
    "information. Do not summarize away a detail that could change what "
    "the reader decides to do next -- when unsure whether a line matters, "
    "keep it. Output only the compressed receipt, no commentary."
)

_REDUCER_USER = "Raw tool output ({n_chars} characters):\n\n{text}\n\nCompressed receipt:"


class EvidenceReducer:
    """Compresses one large tool observation via ``model``, preserving
    evidence rather than keeping a fixed prefix.

    ``min_chars`` is the only real judgement call here: below it, a model
    call to save a few hundred characters costs more than it is worth, and
    flat truncation would not have cut anything off anyway.
    """

    def __init__(
        self,
        model: BaseChatModel,
        min_chars: int = 2000,
        max_output_chars: int = 800,
    ) -> None:
        self.model = model
        self.min_chars = int(min_chars)
        self.max_output_chars = int(max_output_chars)
        self.n_calls = 0
        self.n_fallbacks = 0
        self.chars_saved = 0

    def reduce(self, text: str) -> str:
        """``text``, compressed if it is worth compressing.

        Falls back to plain truncation -- never raises -- on any model
        failure, matching this project's own standing rule that an
        infrastructure hiccup costs the step it happened on, not the
        episode.
        """
        if len(text) < self.min_chars:
            return text

        try:
            compressed = invoke_text(
                self.model,
                [
                    SystemMessage(content=_REDUCER_SYSTEM),
                    HumanMessage(
                        content=_REDUCER_USER.format(n_chars=len(text), text=text)
                    ),
                ],
            ).strip()
        except Exception:
            self.n_fallbacks += 1
            return text[: self.max_output_chars]

        if not compressed:
            self.n_fallbacks += 1
            return text[: self.max_output_chars]

        self.n_calls += 1
        self.chars_saved += max(0, len(text) - len(compressed))
        return compressed[: self.max_output_chars] if len(compressed) > self.max_output_chars else compressed


__all__ = ["EvidenceReducer"]
