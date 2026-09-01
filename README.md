# Meta-Evolver

**A LangGraph engine for LLM agents that get better at a benchmark, generation over generation.**

Most agent frameworks give you a loop that runs a task. This one gives you a loop that runs a task, learns from how it went, and runs it again better — and keeps doing that until the improvement stops.

```bash
pip install -e .
export GEMINI_API_KEY=...

meta-evolver benchmarks                       # what can I run against
meta-evolver evolve --benchmark devops -g 5   # improve for five generations
meta-evolver ablate --memory memories.jsonl   # prove the memory is worth it
```

No key handy? `python examples/offline_demo.py` runs the entire system against a scripted model in about a second.

---

## The idea

An agent that solves a task and forgets it is a very expensive stateless function. Three things have to change between one attempt and the next for the agent to actually improve, and Meta-Evolver evolves all three, in one loop:

| Channel | What changes | Why it alone is not enough |
|---|---|---|
| **Memory** | Failures and successes distil into reusable strategies | A bank that only grows fills its retrieval slots with paraphrases and never unlearns a bad lesson |
| **Prompt** | Failure traces drive OPRO-style rewrites of the system instruction | An unvalidated rewrite each generation is *drift*, not improvement |
| **Curriculum** | The environment gets harder as the agent gets better | A fixed benchmark stops teaching the moment it is solved |

They are coupled on purpose. A harder curriculum produces richer failures → richer failures produce better memories and sharper prompts → better memories and prompts clear the next difficulty. That coupling is the *meta* in Meta-Evolver.

---

## Two graphs

### The episode graph — one rollout

```
  START → prepare → think → route
                      ↑       ├── (prose, no tool call) → nudge ──┐
                      │       └── (tool call) → act → adapt       │
                      └────────────── continue ───────────────────┘
                                          │
                            (done / budget) → finalize → END
```

`adapt` is the node that earns the structure. It runs after every action and owns the whole exploration policy: stagnation detection, memory eviction, and the state-exhaustion fallback. As a node rather than a branch inside a `while` loop, it is testable without an LLM, inspectable mid-episode, and checkpointable.

### The evolution graph — one generation

```
  START → sample_tasks ─(Send fan-out)→ rollout → score → induce → credit → prune
                ↑                                                              │
                │                                                              ▼
                └──── (continue) ── checkpoint ← curriculum ← optimize_prompt ─┘
                                          │
                            (converged / budget) → END
```

Rollouts fan out concurrently with LangGraph's `Send`; `score` is the barrier that waits for all of them.

---

## What makes the loop actually converge

Most of the design here is about *not* fooling yourself. Five decisions do the work:

**Memories are credited, not just collected.** Every episode records which memory ids were in its prompt. After the generation, each gets `uses += 1` and `wins += 1` if the episode succeeded — a Beta posterior over "episodes citing this succeed". Retrieval ranks by similarity *modulated by* that utility, so a memory that keeps appearing alongside failures loses its slot before it is ever deleted. `prune` then removes items that have had a fair trial and lost.

**Prompt candidates must earn adoption.** The optimizer proposes; a held-out validation split decides. A candidate is adopted only if it beats the incumbent by a margin. The validation set is stable across generations and never enters training, so a pass rate at generation 4 is comparable with one at generation 1.

**Infrastructure errors are not task failures.** A rate-limited episode has `error` set and `usable == False`, and every downstream learner skips it. Scoring an API timeout as a failure poisons the memory bank, the prompt optimizer and the curriculum at once.

**Regressions are counted, not averaged away.** A generation that fixes two tasks and breaks two looks identical to one that changed nothing. Since every channel rewrites shared state, churn of exactly that kind is the expected failure mode — so it is reported.

**Everything stochastic is seeded reproducibly.** Fault injection, observation noise, task sampling and layout shuffling all derive from a stable digest, not `hash()` (randomized per process) or a tuple seed (a `TypeError` since Python 3.11). Without this, an A/B between generations measures RNG drift.

---

## Retrieved memory as a trap, and the fix

Retrieval-augmented agents have a specific, reproducible failure mode on out-of-distribution tasks. The bank returns the nearest strategy; the strategy does not apply; the agent follows it anyway, because a confident instruction in the system prompt outweighs a few discouraging observations. It then loops — re-checking the places the memory named, growing more certain with each empty result. **More retrieval makes this worse**: the same wrong prior is re-injected every turn.

`AdaptiveExplorationController` applies three coupled mechanisms:

1. **Soft priors.** Memories are framed as fallible hints with an explicit escape clause, never as instructions.
2. **Stagnation eviction.** After `patience` steps with no reward increase *and* no newly-visited entity, the memory block is dropped from the prompt. Novelty counts as progress deliberately: in sparse-reward environments a reward-only detector either fires constantly or never in time.
3. **State-exhaustion fallback.** Eviction removes a bad prior but leaves nothing behind, and an agent with no prior repeats itself — so the freed space is filled with what the tracker knows: visited entities, unvisited candidates, breadth-first instructions.

Eviction is one-way within an episode. Re-admitting a prior that already failed restarts the loop it was introduced to break.

The `textgame` benchmark's eval split exists to test this: its layouts put the target where a kitchen-search prior would look last.

---

## Plugging in your own benchmark

One class, three methods, one decorator:

```python
from meta_evolver import BenchmarkAdapter, register_benchmark

@register_benchmark("my-suite")
class MySuite(BenchmarkAdapter):
    def task_ids(self, split="train"):
        return load_ids(split)

    def make_env(self, task_id, curriculum_level=0.0, seed=0):
        return MyEnv(task_id)          # implements ActionableEnv

    def instruction_for(self, task_id):
        return load_instruction(task_id)
```

Everything else — retrieval, adaptive control, induction, credit assignment, prompt evolution, the curriculum — applies unchanged.

Three shortcuts, if you would rather not write an environment:

```python
# 1. Tasks are Python functions + a verifier over the call log.
from meta_evolver.benchmarks.custom import FunctionBenchmark, Task

def search(query: str) -> dict:
    """Search the incident index."""      # the docstring becomes the tool description
    return {"hits": index.search(query)}

bench = FunctionBenchmark(
    name="triage",
    tools={"search": search, "answer": lambda text: {"answer": text}},
    tasks=[Task(id="t1", instruction="Which release broke SSO?",
                verify=lambda calls: any("4.2.1" in str(c.result) for c in calls),
                terminal_tools=("answer",))],
)

# 2. You already have a Gym-shaped or EnvHarness-shaped environment.
from meta_evolver.benchmarks.external import ExternalEnvAdapter, TextEnvAdapter
env = ExternalEnvAdapter(my_env)         # duck-typed; no shared base class needed
env = TextEnvAdapter(my_text_game)       # string actions get a `do(text=...)` tool
```

See [`examples/custom_benchmark.py`](examples/custom_benchmark.py) for the whole integration in one file.

---

## Built-in benchmarks

Both are self-contained, deterministic, and run offline — no dataset download, no simulator install, so a behaviour regression shows up as a failing test rather than a drifting number nobody can reproduce.

**`devops`** — five production incidents with verifiable ground truth. Each punishes a different shortcut. `rate_limit` and `disk_pressure` are the interesting ones: the obvious fix is *wrong*, and taking it scores zero. They exist because an agent can pass the first three by pattern-matching "find the number, make it bigger".

**`textgame`** — ALFWorld-style embodied search. The eval split is adversarial to memory priors, as described above.

---

## Provider handling

Any [litellm](https://github.com/BerriAI/litellm) model id works: `--model openai/gpt-4.1`, `--model anthropic/claude-opus-4-7`, `--model gemini/gemini-3-flash` (default).

One provider quirk is handled centrally rather than at each call site: **Google removed the manual sampling overrides from the Gemini API.** `temperature`, `top_p` and `top_k` are deprecated there — generation is steered by the thinking level instead. `LLMClient._prepare` strips them for any Gemini route (the `gemini/` prefix, `vertex_ai/gemini-*`, and bare `gemini-*` names) while leaving them intact for providers that still honour them. A config carrying `temperature: 0.4` stays correct on both.

---

## Layout

```
meta_evolver/
  core/        types, ActionableEnv, Rules harness, registry, seeding
  graphs/      episode.py, evolution.py, state.py      ← the LangGraph engine
  memory/      bank.py (MMR + credit + prune), induction.py
  adaptive/    controller.py, tracker.py               ← OOD mitigation
  prompts/     optimizer.py (OPRO), templates.py
  harness/     curriculum.py                           ← difficulty escalation
  benchmarks/  base.py, devops.py, textworld.py, custom.py, external.py
  tools/       routing.py                              ← tool governance
  llm/         client.py, embeddings.py
  telemetry/   engine.py
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q          # 86 tests, ~3s, no network
```

The whole engine runs against `ScriptedLLMClient`, so the evolution loop, memory curation, curriculum escalation and prompt selection are all covered without an API key.

---

## Grounding

The design follows the taxonomy in [Self-Improvements in Modern Agentic Systems: A Survey](https://arxiv.org/abs/2607.13104) (Ren et al., 2026), which splits self-improvement into foundation-model updates and *scaffolding* updates — prompt, memory, and tool governance. Meta-Evolver is a scaffolding-improvement system across all three, plus environment curriculum.

Memory induction follows [ReasoningBank](https://arxiv.org/abs/2509.25140) (Google Cloud AI Research, 2025): distil generalizable strategies from both successes *and* failures, rather than storing raw trajectories or only successful routines.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full mapping, including what this implementation adds and what it deliberately leaves out.

---

## Licence

Apache 2.0. See [LICENSE](LICENSE).
