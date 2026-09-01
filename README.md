# Meta-Evolver

**A LangGraph engine for LLM agents that get better at a benchmark, generation over generation.**

Most agent frameworks give you a loop that runs a task. This one gives you a loop that runs a task, learns from how it went, and runs it again better — and keeps doing that until the improvement stops.

```bash
pip install -e ".[google]"     # or [openai], [anthropic], [all]
export GEMINI_API_KEY=...

meta-evolver benchmarks                       # what can I run against
meta-evolver evolve --benchmark devops -g 5   # improve for five generations
meta-evolver ablate --memory memories.jsonl   # prove the memory is worth it
```

No key handy? `python examples/offline_demo.py` runs the entire system against a scripted model in about a second.

Want the memory bank in a real database rather than a file?

```bash
docker compose -f docker/docker-compose.yml up -d postgres
export META_EVOLVER_DB_URL=postgresql://meta:meta@localhost:5433/meta_evolver
```

---

## The idea

An agent that solves a task and forgets it is a very expensive stateless function. Three things have to change between one attempt and the next for the agent to actually improve, and Meta-Evolver evolves all three, in one loop:

| Channel | What changes | Why it alone is not enough |
| --- | --- | --- |
| **Memory** | Failures and successes distil into reusable strategies | A bank that only grows fills its retrieval slots with paraphrases and never unlearns a bad lesson |
| **Prompt** | Failure traces drive OPRO-style rewrites of the system instruction | An unvalidated rewrite each generation is *drift*, not improvement |
| **Curriculum** | The environment gets harder as the agent gets better | A fixed benchmark stops teaching the moment it is solved |

Set `--rollouts-per-task 3` and each task is attempted several times; attempts
that *disagree* become contrastive evidence for induction, which is the
synthesis half of [MaTTS](https://arxiv.org/abs/2509.25140). Scoring switches
to pass@K and the report says which K it used, because a pass@3 number is not
comparable with a pass@1 one.

They are coupled on purpose. A harder curriculum produces richer failures → richer failures produce better memories and sharper prompts → better memories and prompts clear the next difficulty. That coupling is the *meta* in Meta-Evolver.

---

## Two graphs

### The episode graph — one rollout

```text
  START → prepare → think → route
                      ↑       ├── (prose, no tool call) → nudge ────────┐
                      │       ├── (failed assertion) → assert_retry ────┤
                      │       └── (valid tool call) → act → adapt ──────┤
                      └───────────────── continue ──────────────────────┘
                                            │
                              (done / budget) → finalize → END
```

`adapt` is the node that earns the structure. It runs after every action and owns the whole exploration policy: stagnation detection, memory eviction, and the state-exhaustion fallback. As a node rather than a branch inside a `while` loop, it is testable without an LLM, inspectable mid-episode, and checkpointable.

### The evolution graph — one generation

```text
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

## Where the memory lives

A memory bank is state that outlives a run, is written concurrently, and is
queried by similarity. A JSONL file handles one of those, which is why storage
is pluggable — pick a backend with a URL and nothing else changes.

| Backend | URL | What it adds |
| --- | --- | --- |
| **File** (default) | `memories.jsonl` | Nothing to install. Atomic replace on write, so an interrupted save cannot destroy the bank. |
| **Postgres + pgvector** | `postgresql://…` | Atomic credit increments, server-side ANN, one bank shared across runs. |
| **MongoDB** | `mongodb://…` | Durability and safe concurrent writes. Vector search only on Atlas; otherwise it says so and the bank scores in Python. |

```bash
meta-evolver evolve --db-url postgresql://meta:meta@localhost:5433/meta_evolver
# or set $META_EVOLVER_DB_URL once and every entry point picks it up
```

Two reasons the database is not cosmetic:

**Credit assignment is a race.** Rollouts fan out concurrently and all credit the same memories. Under a read-modify-write, increments are lost, utilities drift low, and the pruner deletes memories that were doing fine — and the symptom is "the bank stopped improving", not an error. In SQL it is `uses = uses + 1`, and the race does not exist.

**Retrieval stops being O(n).** Embeddings live in a `vector` column with an HNSW cosine index. The query pulls a candidate *pool* rather than exactly `k`, because MMR re-ranks it and diversity needs alternatives to choose between.

**Runs are resumable.** With Postgres configured, LangGraph checkpoints every
superstep, so a run killed in generation four resumes rather than restarting.
Resuming is opt-in: `run_id` is unique per run unless you name one, and reusing
a name is how you ask to continue it.

Memories are scoped by `namespace` (the benchmark name), so one database serves several. That scoping is in the primary key, not just the `WHERE` clause: a memory's id is derived from its text, so two benchmarks that independently learn the same lesson derive the same id — and keyed on the id alone, the second write lands in the first's namespace, where its author cannot read it.

---

## Watching the scaffold evolve

The loop rewrites three pieces of shared state, and the questions you actually
have about a run are about *provenance*: which episodes produced the memory
that later carried a task; whether the prompt adopted in generation 3 helped in
4, or whether a memory pruned at the same time explains it; what changed
between the generation that passed a task and the one that broke it.

Those are path queries. Recorded to Neo4j they are one Cypher line each — and
in the browser, a picture that grows while the run happens.

```bash
docker compose -f docker/docker-compose.yml --profile graph up -d neo4j
export META_EVOLVER_GRAPH_URL=bolt://neo4j:evolution@localhost:7688

meta-evolver evolve --benchmark devops -g 5     # prints the run id and browser URL
meta-evolver graph list                          # the saved queries
meta-evolver graph memory_provenance --run-id run-a1b2c3d4
meta-evolver graph regressions --run-id run-a1b2c3d4 --cypher   # paste into the browser
```

Six node labels, and every edge means *caused / produced / was used by*:

```text
(:Run)-[:HAS_GENERATION]->(:Generation)-[:RAN]->(:Episode)-[:ON_TASK]->(:Task)
(:Generation)-[:USED_PROMPT]->(:Prompt)   (:Generation)-[:AT_LEVEL]->(:CurriculumLevel)
(:Episode)-[:RETRIEVED]->(:Memory)        ← the memory was in the prompt
(:Episode)-[:INDUCED]->(:Memory)          ← the episode produced it
(:Generation)-[:PRUNED]->(:Memory)        (:Prompt)-[:PROPOSED_FROM]->(:Prompt)
```

`RETRIEVED` and `INDUCED` are the pair that make the bank's history legible:
follow both directions from a memory and you have where it came from and
whether it ever helped. Rejected prompt candidates are recorded too — they are
the evidence that alternatives were tried and measured.

It is observability, never machinery: a Neo4j that is down costs you the
picture and nothing else. Every write is guarded and the failure count is
reported.

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

## Advanced Scaffolding Subsystems (API LLM / Frozen-Weight Optimization)

Meta-Evolver operates entirely at the **scaffolding layer**, enabling continuous self-improvement across API-only LLMs without modifying foundation model weights:

1. **In-Flight Assertions & Backtracking (`ScaffoldAssert`)**
   - Intercepts ill-formed tool calls, invalid schemas, and domain constraint violations *before* execution.
   - Triggers intra-step feedback loops (`assert_retry` node) to allow the agent to correct actions without advancing the environment step counter.

2. **Feedback-Driven Multi-Component Reflection (`GEPAPromptOptimizer`)**
   - Decomposes the prompt into modular components (`planning`, `tool_policy`, `error_recovery`, `core_role`).
   - Tracks a population across multiple tasks and samples candidate parents from the non-dominated **Pareto frontier**.
   - Supports component-level reflective mutations and crossover merges between Pareto parents.

3. **Sandbox Variable REPL & Context Navigation (`ScaffoldRLM`)**
   - Separates **Variable Space** from **Token Space** for massive observations (long logs, large DOM trees, dataframes).
   - Injects lightweight `VariableDescriptor` metadata into the prompt; the agent executes sandboxed Python code to slice data and delegates sub-queries to `llm_query` with strict call budgets.

4. **Dynamic Code Scaffolding Evolution (`FlexScaffold`)**
   - Moves executable Python module code into the optimizable parameter space (`FlexModule`).
   - Compiles and runs dynamically synthesized tools and `FlexRule` harness wrappers in isolated execution sandboxes with crash resilience.

5. **OpenTelemetry & MLflow Visual Tracing (`TelemetryTracer`)**
   - Step-level hierarchical span tracking recording node latencies, token consumption, and intermediate events.
   - Exports directly to standard **MLflow Trace UI** format and **OpenTelemetry JSON** for observability in local or cloud dashboards.

---

## Built-in benchmarks

Both are self-contained, deterministic, and run offline — no dataset download, no simulator install, so a behaviour regression shows up as a failing test rather than a drifting number nobody can reproduce.

**`devops`** — five production incidents with verifiable ground truth. Each punishes a different shortcut. `rate_limit` and `disk_pressure` are the interesting ones: the obvious fix is *wrong*, and taking it scores zero. They exist because an agent can pass the first three by pattern-matching "find the number, make it bigger".

**`textgame`** — ALFWorld-style embodied search. The eval split is adversarial to memory priors, as described above.

---

## Models and embeddings

Chat models are LangChain `BaseChatModel` instances built through `init_chat_model`, so any provider integration works: `--model openai:gpt-4.1`, `--model anthropic:claude-opus-4-7`, `--model google_genai:gemini-3-flash` (default). The older `provider/model` spelling resolves too, so existing configs keep working.

LangChain rather than a generic gateway because LangGraph is the orchestration layer here: state carries real `AnyMessage` objects under the `add_messages` reducer, tool calls arrive already normalized on `AIMessage.tool_calls`, and a test double is just another `BaseChatModel`. Nothing in this package re-implements a provider's wire format.

**Embeddings default to `gemini-embedding-2` at 768 of its 3072 dimensions.**

Both Gemini embedding models are Matryoshka-trained — the most significant structure is packed into the leading dimensions, so a truncated vector keeps nearly all its retrieval signal at a quarter of the storage and a quarter of the dot-product cost. Both costs are real: a bank persists every vector to JSONL, and MMR retrieval is O(k·n) dot products per episode.

The reason for `-2` specifically is normalization. It renormalizes truncated output automatically; `gemini-embedding-001` does not, so a 768-dim vector from `-001` is no longer unit-norm and every consumer has to renormalize it or silently start comparing by magnitude as well as direction. `--embed-model` and `--embed-dimensions` override both.

Queries and stored memories are embedded **asymmetrically** — `RETRIEVAL_QUERY` for what searches the bank, `RETRIEVAL_DOCUMENT` for what goes into it. They project into the same space from different sides, and using the document side for both discards signal the model was trained to provide.

Embeddings are a *separate* model from the chat model, on purpose: a scripted or local chat model must not disable retrieval, and switching chat provider must not re-embed an existing bank into a different vector space.

**One provider quirk is handled centrally** rather than at each call site: Google removed the manual sampling overrides from the Gemini API. `temperature`, `top_p` and `top_k` are deprecated there — generation is steered by the thinking level instead. `build_chat_model` strips them for any Gemini route (`gemini/`, `google_genai:`, `vertex_ai/gemini-*`, bare `gemini-*`) while leaving them intact for providers that still honour them. A config carrying `temperature: 0.4` stays correct on both.

---

## Layout

```text
meta_evolver/
  core/        types, ActionableEnv, Rules harness, flex (dynamic code), registry, seeding
  graphs/      episode.py (assertions + backtracks), evolution.py, state.py
  memory/      bank.py (MMR + credit + prune), induction.py
  adaptive/    controller.py, tracker.py               ← OOD mitigation
  prompts/     optimizer.py (OPRO), gepa.py (multi-component Pareto), templates.py
  harness/     curriculum.py                           ← difficulty escalation
  benchmarks/  base.py, devops.py, textworld.py, custom.py, external.py
  tools/       routing.py, assertions.py, repl.py (ScaffoldRLM)
  llm/         client.py, embeddings.py                ← LangChain models
  storage/     jsonl.py, postgres.py, mongo.py       ← pluggable persistence
               checkpoint.py                          ← resumable runs
  graph_view/  recorder.py, schema.py                 ← the causal graph
  telemetry/   engine.py, tracer.py (OTel / MLflow traces)
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q          # 165 passed tests, no network; DB tests skip if nothing is listening
```

To exercise the database backends too:

```bash
docker compose -f docker/docker-compose.yml up -d postgres
docker compose -f docker/docker-compose.yml --profile mongo up -d mongo
docker compose -f docker/docker-compose.yml --profile graph up -d neo4j
pytest -q          # storage contract across all three, plus the causal graph
```

The whole engine runs against `ScriptedChatModel` — a real `BaseChatModel`, so the graphs exercise exactly the path a live model takes (`bind_tools`, `AIMessage.tool_calls`, `ToolMessage` round-tripping) rather than a parallel mock path that can drift from it. The evolution loop, memory curation, curriculum escalation and prompt selection are all covered without an API key.

---

## Grounding

The design follows the taxonomy in [Self-Improvements in Modern Agentic Systems: A Survey](https://arxiv.org/abs/2607.13104) (Ren et al., 2026), which splits self-improvement into foundation-model updates and *scaffolding* updates — prompt, memory, and tool governance. Meta-Evolver is a scaffolding-improvement system across all three, plus environment curriculum.

Memory induction follows [ReasoningBank](https://arxiv.org/abs/2509.25140) (Google Cloud AI Research, 2025): distil generalizable strategies from both successes *and* failures, rather than storing raw trajectories or only successful routines.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full mapping, including what this implementation adds and what it deliberately leaves out.

---

## Licence

Apache 2.0. See [LICENSE](LICENSE).
