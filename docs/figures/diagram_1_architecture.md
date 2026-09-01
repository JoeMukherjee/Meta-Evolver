# Diagram 1: Meta-Evolver Dual-Graph System Architecture

## Scientific Specification & Visualization Logic

### Purpose & Research Context
Visualizes the overall system architecture of **Meta-Evolver** as an API-only, scaffolding-level self-improving agent engine operating over two coupled LangGraph state machines without modifying foundation model weights $\theta$.

---

### Structural Components & Flow

#### 1. Top Section: Outer Evolution Graph (One Generation)
* **`sample_tasks`**: Disjointly partitions the benchmark suite into training tasks and held-out validation tasks.
* **`Send(fan-out)`**: Asynchronous rollout concurrency executing $K$ stochastic attempts per task (MaTTS test-time scaling).
* **`score`**: Synchronization barrier collecting completed rollout trajectories.
* **`induce`**: Contrastive strategy distillation across disagreeing trajectories $\to$ extracts reusable positive strategies and anti-pattern recovery rules into `ReasoningMemoryBank`.
* **`credit` & `prune`**: Bayesian Beta posterior update:
  $$\text{Utility}(m) = \frac{\text{wins}(m) + 1.0}{\text{uses}(m) + 2.0}$$
  Prunes underperforming strategies with $\text{uses} \ge 4$ and $\text{utility} < 0.34$.
* **`optimize_prompt`**: Multi-component GEPAPromptOptimizer sampling parents from the multi-objective **Pareto Frontier** across task categories, proposing mutated components with held-out validation gating ($\Delta \ge +5\%$).
* **`adapt_curriculum`**: Hysteretic difficulty escalation ($0.20 \to 0.40 \to 0.60 \to 0.80$) adding transient faults, distractor noise, verification gates, and tighter step limits.

#### 2. Bottom Section: Inner Episode Graph (Single Rollout)
* **`prepare`**: Resets environment, computes asymmetric query embedding via `gemini-embedding-2` (768 dims), retrieves top-$k$ memories via MMR.
* **`think`**: Dynamic system prompt assembly (`render_system_prompt`), tool filtering via `ToolRouter`, LLM call.
* **`assert_retry`**: Intra-step semantic verification (`ScaffoldAssert`). If an action violates constraints, feedback is injected without consuming environment steps.
* **`act`**: Executes environment step, records duration, step record, and tool payload.
* **`adapt`**: Updates `AdaptiveExplorationController`. If stagnation $\ge 6$ steps without reward or novel entity discovery, evicts misleading prior and switches to state-exhaustion breadth-first search.
* **`finalize`**: Ground-truth verifier scoring, infrastructure error isolation (`usable=False` on API drops).

---

### Visual Style Directive for Gemini Image Generation
* **Aesthetic**: Academic journal vector diagram, pure white background (`#FFFFFF`), crisp high-contrast lines.
* **Color Palette**: Royal Blue (`#2980B9`) for Outer Loop nodes, Emerald Green (`#27AE60`) for Inner Loop nodes, Amethyst Purple (`#8E44AD`) for Memory & Bayesian Crediting, Coral Red (`#E74C3C`) for Assertion & Eviction pathways.
* **Typography**: Clean sans-serif labels, crisp boxes with subtle drop shadows, clean directional arrows with explicit branch labels.
