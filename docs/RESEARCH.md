# Research grounding

What Meta-Evolver borrows, what it adds, and what it deliberately leaves out.

---

## The frame

[**Self-Improvements in Modern Agentic Systems: A Survey**](https://arxiv.org/abs/2607.13104) — Ren, Guo, Rong, Chen, Wang, Li, Yang, Zhuge, Schmidhuber et al. (Jilin University / KAUST / IDSIA, July 2026)

The survey formalises a modern agent as `A = (θ, Σ)`: foundation-model parameters `θ` plus an operational **scaffold** `Σ = (prompts, memory, tools, control logic)`. Self-improvement is a self-induced update operator applied to one or the other. That split is the cleanest available account of what this project is:

> **Meta-Evolver is a scaffolding-improvement system.** It never touches `θ`. It updates prompts, memory, and tool exposure — plus one channel the survey's taxonomy does not cover, the environment itself.

The survey's taxonomy of scaffolding improvement maps onto the codebase directly:

| Survey category | Meta-Evolver |
|---|---|
| Prompt optimization (`p → p'`) | [`prompts/optimizer.py`](../meta_evolver/prompts/optimizer.py) — OPRO with held-out selection |
| Memory evolution (`m → m'`) | [`memory/bank.py`](../meta_evolver/memory/bank.py) — vector retrieval, MMR, credit-driven CRUD |
| Tool governance (`T → T'`) | [`tools/routing.py`](../meta_evolver/tools/routing.py) — dynamic tool routing |
| Full scaffolding update (`Σ → Σ'`) | *not implemented* — see [Deliberate omissions](#deliberate-omissions) |
| — | [`harness/curriculum.py`](../meta_evolver/harness/curriculum.py) — environment escalation |

Three of the survey's recommendations shaped decisions here that would otherwise have gone the easy way:

**"Confine updates to the scaffold when feedback is noisy."** Everything here is scaffold-level and reversible. Nothing is distilled into weights, so a bad generation costs a prompt revert rather than a training run.

**"Governed criticism — decoupled evaluators, to prevent reward hacking and self-confirming loops."** The prompt optimizer proposes but does not score. A held-out split does, and it never enters training. Without that separation, each generation adopts a rewrite nobody measured, and by generation five the agent is running an instruction that was never shown to be better than the one it replaced.

**"Report full performance trajectories, transfer on held-out data, and regression rates."** `GenerationReport` carries all three. Regressions in particular: a generation that fixes two tasks and breaks two has the same pass rate as one that changed nothing, and since every channel here rewrites shared state, that churn is the *expected* failure mode.

---

## Memory

[**ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory**](https://arxiv.org/abs/2509.25140) — UIUC / Yale / Google Cloud AI Research (Sept 2025, ICLR 2026)

The paper's core finding is that *what* you store dominates *how much*: distilled, generalizable reasoning strategies beat both raw trajectories and successful-routine libraries. And critically, failures are as informative as successes — an agent that only learns from what worked never learns what does not.

Adopted in [`memory/induction.py`](../meta_evolver/memory/induction.py):

- Strategies, not trajectories. The induction prompt explicitly contrasts a transferable lesson against an episode-specific one.
- Both polarities. `polarity="failure"` items are anti-patterns and render differently in the prompt, so the agent reads them as "do not" rather than as a procedure.
- Batched induction over an episode *set*. Reflecting on one trajectory produces lessons that restate that task; reflecting on several forces the model to find what they share, which is the part that transfers.

The paper also introduces **MaTTS** (memory-aware test-time scaling): more rollouts per task give richer contrastive signal for synthesising memory, and better memory guides more effective scaling. Meta-Evolver's fan-out gives the mechanism for this — `Send` already parallelises rollouts — but the current loop runs one rollout per task per generation. Multi-rollout contrastive induction is the obvious next step and is not implemented.

### What this adds beyond ReasoningBank

**Credit assignment.** ReasoningBank distils and retrieves; it does not track whether an individual memory earns its retrieval slot. Here, every trajectory records `retrieved_memory_ids`, and after each generation those memories are charged for the outcome — `uses += 1`, `wins += 1` on success. `utility` is the Beta(1,1) posterior mean, retrieval ranks by similarity blended with it, and `prune` removes items that have had a fair trial (`min_uses`) and lost.

This closes the loop the survey calls signal-driven memory processing (Create/Read/**Update**/**Delete**). Without it, the bank is append-only: it plateaus, and a confidently-wrong lesson is re-injected into every similar task forever.

**Dedup with record inheritance.** A near-duplicate merges into the incumbent rather than taking a second slot, and inherits its track record — so a strategy that keeps being rediscovered is not repeatedly re-tried from a blank slate.

**MMR retrieval.** With a bank of near-synonymous lessons, pure cosine returns five paraphrases of one idea. MMR spends the slots on genuinely different strategies, which matters most exactly when the task is unfamiliar.

---

## The OOD failure mode

Related work converged on the same problem from several directions:

- [**SkillOS: Learning Skill Curation for Self-Evolving Agents**](https://arxiv.org/abs/2605.06614) (May 2026) — curation, not accumulation, is the bottleneck.
- [**MemEvolve: Meta-Evolution of Agent Memory Systems**](https://arxiv.org/abs/2512.18746) (OPPO / LV-NUS, Dec 2025) — evolving the memory *architecture*, not only its contents.
- [**EXG: Self-Evolving Agents with Experience Graphs**](https://arxiv.org/abs/2605.17721) and [**HyperSkill**](https://arxiv.org/abs/2608.16114) (2026) — graph- and hypergraph-structured skill memory.

Meta-Evolver's [`adaptive/controller.py`](../meta_evolver/adaptive/controller.py) addresses the specific pathology these all circle: **on an out-of-distribution task, retrieval actively hurts.** The bank returns the nearest strategy, it does not apply, and the agent follows it anyway — a confident instruction in the system prompt outweighs a few discouraging observations. It then loops, growing more certain with each empty result. More retrieval makes it worse, because the same wrong prior is re-injected every turn.

The mitigation is three coupled mechanisms — soft-prior framing, stagnation eviction, and a state-exhaustion fallback — documented in the module and in the README. Two details are load-bearing and easy to get wrong:

**Novelty counts as progress.** A reward-only stagnation detector is useless in sparse-reward environments, where reward fires only at the terminal step. It would either evict constantly or, with generous patience, never in time to matter.

**Eviction is one-way within an episode.** Re-admitting a prior that already failed restarts the loop eviction exists to break.

The `textgame` eval split is built to trigger the pathology: layouts place the target where a kitchen-search prior looks last. That makes the controller's contribution measurable rather than assumed — `meta-evolver ablate` reports it.

---

## Prompt evolution

OPRO — optimization by prompting — is well-established: show a meta-model the current instruction, its measured score, and what went wrong; ask for a better one. The survey groups it with population-based and textual-gradient methods under prompt optimization.

The mechanism works. What decides whether it *helps* over many generations is what the survey calls governed criticism, and it is the part most implementations skip:

> **A proposed prompt is a hypothesis, not an improvement.**

So [`optimizer.py`](../meta_evolver/prompts/optimizer.py) proposes several candidates and returns them for measurement; the evolution graph runs each on the held-out split and adopts a winner only if it clears the incumbent by a margin. Three smaller guards, each from watching this fail:

- Candidates that drop `{memory_section}` or `{guidance_section}` are **repaired, not discarded** — a good prompt that forgot a placeholder is worth keeping, and one that silently loses them disables retrieval while still scoring as an improvement.
- Failure traces are sampled across **distinct tasks**; five traces of one pathological task produce a prompt overfitted to it.
- Failure *signatures* (repeat-action loops, budget exhaustion, blocked submissions) are **computed, not asked for**. They are facts the meta-model would otherwise infer from truncated traces, and it infers them unreliably.

---

## Curriculum

Not a category in the survey's taxonomy — its scope is the agent, not the environment — but it addresses a problem the survey names directly in its discussion of evaluation: measuring *continuous* improvement.

A fixed benchmark stops teaching the moment it is solved. Every subsequent generation runs the same tasks, produces the same successes, and the memory bank and prompt optimizer have nothing left to learn from. The curve flattens, and it is easy to mistake that plateau for convergence.

[`harness/curriculum.py`](../meta_evolver/harness/curriculum.py) makes difficulty a variable. `level ∈ [0, 1]` derives a harness stack: transient faults, then a verification gate, then distractor observations, then a tighter step budget. Each band adds a *distinct* failure mode rather than turning one dial up — an agent that has learned to retry has not thereby learned to verify.

Two details:

- Promotion and demotion are **hysteretic** (promote above 0.7, demote below 0.3). With a single threshold, a run sitting near it oscillates between two difficulties and learns neither.
- Pass rate is always **relative to the current level**, so a flat rate at a rising level is progress. The stall detector in `checkpoint` knows this; a naive one would stop the run exactly when it started working.

This is closest in spirit to EnvHarness-style environment mutation: the `Rules` A/T/O hook design in [`core/rules.py`](../meta_evolver/core/rules.py) is adapted from it, so a harness layer is itself an `ActionableEnv` and the agent cannot tell a raw benchmark from one under three layers of perturbation.

---

## Deliberate omissions

Things the literature supports that are **not** here, and why.

**Parametric updates.** SEED ([2607.14777](https://arxiv.org/abs/2607.14777)), Latent OPSD ([2608.13040](https://arxiv.org/abs/2608.13040)) and Evolving-RL ([2605.10663](https://arxiv.org/abs/2605.10663)) distil experience into weights. That is the survey's "slow loop", and its own design guidance is to defer parametric consolidation until new behaviours are stable. Scaffold-only keeps every update reversible and every run reproducible from a seed.

**Full scaffolding update / self-modifying code.** The survey's deepest intervention tier: the agent rewrites its own control logic. It is also where its safety discussion is most emphatic — treat the agent as untrusted code, enforce layered gating. Not something to add without a sandbox story.

**Graph-structured memory.** EXG and HyperSkill link memories into experience graphs and hypergraphs. The flat bank with MMR here is a deliberate first cut: the graph variants' benefit shows up at bank sizes well past what a five-task benchmark produces, and an unevaluated graph is harder to debug than a flat list. `MemoryItem` carries `scenario` and `source_task_ids`, which is the edge data a graph layer would need.

**MaTTS.** Discussed above — the fan-out exists, the contrastive induction does not.

### One risk worth naming

[**Benign Alone, Harmful Together: Exploiting Experience Composition in Self-Evolving LLM Agents**](https://arxiv.org/abs/2608.01759) (Aug 2026) shows that individually-harmless memories can compose into harmful behaviour — a real attack surface for any system that persists distilled experience and re-injects it.

Meta-Evolver's partial mitigations are incidental rather than designed: memories are human-readable and inspectable (`meta-evolver` writes the bank as JSONL you can read), utility-based pruning removes items correlated with bad outcomes, and induction only ever sees the agent's own trajectories. None of that is a defence against a deliberately poisoned bank. **Treat a memory bank from an untrusted source as untrusted input.**

---

## Reading list

| Paper | Relevance |
|---|---|
| [Self-Improvements in Modern Agentic Systems: A Survey](https://arxiv.org/abs/2607.13104) | The frame. Start here. |
| [ReasoningBank](https://arxiv.org/abs/2509.25140) | Memory induction; the direct ancestor of `memory/` |
| [SkillOS](https://arxiv.org/abs/2605.06614) | Curation over accumulation |
| [MemEvolve](https://arxiv.org/abs/2512.18746) | Meta-evolution of the memory architecture itself |
| [EXG: Experience Graphs](https://arxiv.org/abs/2605.17721) | Graph-structured memory |
| [HyperSkill](https://arxiv.org/abs/2608.16114) | Hypergraph skill memory; what/when/how to store |
| [Evo-Harness](https://arxiv.org/abs/2608.15071) | Compiling context into reusable harnesses |
| [Evolving-RL](https://arxiv.org/abs/2605.10663) | End-to-end optimization of the self-evolving capability |
| [SEED](https://arxiv.org/abs/2607.14777) | On-policy distillation for agentic RL |
| [Benign Alone, Harmful Together](https://arxiv.org/abs/2608.01759) | Safety of composed experience |
| [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/graph-api) | `StateGraph`, reducers, `Send`, checkpointing |
| [LangChain models](https://docs.langchain.com/oss/python/langchain/models) | `init_chat_model`, `bind_tools`, message types |
| [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings) | `output_dimensionality`, MRL truncation, `-2` renormalization |
