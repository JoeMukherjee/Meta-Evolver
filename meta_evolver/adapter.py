"""MetaAdapter -- a trained agent's learning, detached from the harness that
trained it, and the mechanism another instance attaches it through.

Two pieces, matched to the two things an agent actually learns:

**The bank travels as data.** :meth:`MetaAdapter.save` always exports the
memory bank to a portable JSONL file next to the manifest -- even when the
source bank is Mongo- or Postgres-backed -- because the whole point of a
detachable artifact is that it does not require the instance that produced
it still being reachable. :meth:`Agent.attach_adapter` merges it through
:meth:`~meta_evolver.memory.bank.ReasoningMemoryBank.extend`'s existing
dedup, with usage counts discounted first -- imported evidence has not been
earned *here* yet.

**The prompt travels as a candidate, never a fact.** SoL-Pi's held-out
separation and this project's own rehearsal gate both exist because a
prompt that looks right is not the same claim as a prompt that measured
right. An attached adapter's prompt is exactly as untrusted as anything
:class:`~meta_evolver.prompts.optimizer.PromptOptimizer` itself proposes --
:class:`SeededOptimizer` is what makes that literally true rather than a
policy nobody enforces: it hands the imported text to the *same*
``_select_prompt`` margin-and-rehearsal gate every other candidate clears,
by injecting it into the wrapped optimizer's very next ``propose()`` call.

The manifest (``manifest.json``) also carries the ``rehearsal_ids`` the
source agent had built up -- a small, real regression oracle travelling with
the artifact, not just the prompt text on its own.

**Adopted tools travel as a record, not as code to re-run.** Every tool an
evolver's tool-self-modification gate adopted (see
:mod:`meta_evolver.tools.proposer`, :mod:`meta_evolver.tools.sandbox`) rides
in the manifest, each tagged ``permissions: "pure_function"`` -- the one
capability class the sandbox gate's AST allowlist actually establishes. An
attaching instance still decides for itself whether to trust and re-register
imported tool source; this module only carries the record, the same way it
carries a prompt candidate without applying it. Every :meth:`save` also
appends one line to ``archive.jsonl`` next to the manifest -- a DGM-style
history of what this path has held over time, adopted tools or not, so a
later look at the same path is not limited to the most recent snapshot.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meta_evolver.core.types import MemoryItem
from meta_evolver.prompts.optimizer import PromptCandidate

if TYPE_CHECKING:
    from meta_evolver.core.evolver import MetaEvolver

#: Bumped only on a breaking change to the manifest shape.
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class MetaAdapter:
    """One agent's learning: its bank, its evolved prompt, and where both
    came from. ``MetaAdapter.load(dir)`` on a directory with no
    ``manifest.json`` -- an artifact from before this module existed --
    returns one with ``legacy=True`` and everything else at its default, so
    an old ``Agent.save()`` output still attaches; it just carries no
    provenance or rehearsal oracle to check against.
    """

    schema_version: int = SCHEMA_VERSION
    model_name: str = ""
    benchmark: str = ""
    """Provenance, not a constraint: what produced this adapter. Attaching
    it to a *different* benchmark is allowed -- that is the whole point of
    the prompt going through validation rather than being trusted -- but the
    caller can inspect this to know it is a cross-benchmark transfer."""
    prompt_text: str = ""
    prompt_version: str = "base"
    curriculum_level: float = 0.0
    rehearsal_ids: list[str] = field(default_factory=list)
    source_generations: int = 0
    source_pass_rate: float | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    """Every tool that cleared both self-modification gates this run --
    see ``EvolutionState.adopted_tools``. Empty for every adapter produced
    before Phase 4, and for any run with ``enable_tool_proposals=False``
    (the default): an old or unaffected manifest round-trips through this
    field with nothing to say."""
    created_at: str = ""
    content_sha256: dict[str, str] = field(default_factory=dict)
    legacy: bool = False

    @classmethod
    def from_evolver(cls, evolver: "MetaEvolver") -> MetaAdapter:
        return cls(
            model_name=evolver.model_name,
            benchmark=evolver.benchmark.name,
            prompt_text=evolver.prompt_template,
            prompt_version=evolver.prompt_version,
            curriculum_level=evolver.curriculum_level,
            rehearsal_ids=list(evolver.rehearsal_ids),
            source_generations=len(evolver.reports),
            source_pass_rate=(evolver.reports[-1].pass_rate if evolver.reports else None),
            tools=[
                {**t, "permissions": "pure_function"} for t in getattr(evolver, "adopted_tools", [])
            ],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def save(self, evolver: "MetaEvolver", path: str | Path) -> dict[str, Path]:
        """Export ``evolver``'s bank and prompt to ``path``, plus this
        manifest. Returns the paths written, including ``"manifest"``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        memory_path = path / "memory.jsonl"
        prompt_path = path / "prompt.txt"
        # An explicit memory_path always forces a portable JSONL export, even
        # for a store-backed bank -- see the module docstring.
        evolver.save(memory_path=memory_path, prompt_path=prompt_path)

        self.content_sha256 = {
            "memory": _sha256_file(memory_path),
            "prompt": _sha256_file(prompt_path),
        }
        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

        archive_path = path / "archive.jsonl"
        with archive_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "created_at": self.created_at,
                        "prompt_version": self.prompt_version,
                        "source_generations": self.source_generations,
                        "source_pass_rate": self.source_pass_rate,
                        "tool_names": [t.get("name", "") for t in self.tools],
                        "content_sha256": self.content_sha256,
                    }
                )
                + "\n"
            )
        return {
            "manifest": manifest_path,
            "memory": memory_path,
            "prompt": prompt_path,
            "archive": archive_path,
        }

    @classmethod
    def load(cls, path: str | Path) -> MetaAdapter:
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return cls(legacy=True)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data.pop("legacy", None)
        adapter = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        prompt_path = path / "prompt.txt"
        if not adapter.prompt_text and prompt_path.exists():
            # An adapter saved before prompt_text was captured in the
            # manifest itself -- fall back to the sidecar file.
            adapter.prompt_text = prompt_path.read_text(encoding="utf-8")
        return adapter

    def memory_items(self, path: str | Path) -> list[MemoryItem]:
        """Parse ``<path>/memory.jsonl`` -- the actual portable bank
        content, separate from the manifest that describes it."""
        memory_path = Path(path) / "memory.jsonl"
        if not memory_path.exists():
            return []
        items = []
        for line in memory_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(MemoryItem(**json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return items


class SeededOptimizer:
    """Wraps a prompt optimizer, injecting one externally-supplied candidate
    into its very next ``propose()`` call.

    This is the entire mechanism that keeps an attached adapter's prompt
    from being applied directly: :meth:`seed` queues it, and the next
    generation's ``optimize_prompt`` node hands it to ``_select_prompt``
    alongside whatever the wrapped optimizer proposes on its own, gated by
    the exact same held-out margin and rehearsal check as any other
    candidate. Consumed once -- a second generation with nothing newly
    seeded behaves as a transparent pass-through to ``inner``.

    The seed is returned *even when the wrapped optimizer proposes nothing*
    (a generation with no failures yet): otherwise a request to evaluate an
    imported prompt could silently vanish on exactly the runs an agent is
    already doing well on, which is precisely when trying an import is
    cheapest to check.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._seed: PromptCandidate | None = None
        self.last_error = ""

    def seed(self, text: str, version: str = "imported") -> None:
        self._seed = PromptCandidate(text=text, version=version, parent="imported")

    def propose(
        self, current_prompt: str, trajectories: Any, generation: int = 0, benchmark: str = ""
    ) -> list[PromptCandidate]:
        candidates = list(
            self.inner.propose(
                current_prompt, trajectories, generation=generation, benchmark=benchmark
            )
        )
        self.last_error = getattr(self.inner, "last_error", "")
        if self._seed is not None:
            seed, self._seed = self._seed, None
            return [seed, *candidates]
        return candidates


def discount_usage(items: list[MemoryItem], factor: float) -> list[MemoryItem]:
    """Scale ``wins``/``uses`` on imported items before they enter a new
    bank -- imported evidence has not been earned in *this* bank yet, and
    should not arrive with the same standing as something proven here.
    ``factor=1.0`` imports full-strength; ``0.0`` imports as brand new
    (unused) items, keeping only the text.
    """
    factor = max(0.0, min(1.0, factor))
    for item in items:
        item.uses = int(round(item.uses * factor))
        item.wins = int(round(item.wins * factor))
        item.domain_stats = {
            domain: [w * factor, u * factor] for domain, (w, u) in item.domain_stats.items()
        }
    return items


__all__ = ["SCHEMA_VERSION", "MetaAdapter", "SeededOptimizer", "discount_usage"]
