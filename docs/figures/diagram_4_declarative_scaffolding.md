# Diagram 4: Declarative Scaffolding & Intra-Step Compiler Architecture

## Scientific Specification & Visualization Logic

### Purpose & Research Context
Details the declarative compiler layer adapted from Stanford's DSPy framework into Meta-Evolver, optimizing the scaffolding around API-only LLMs without touching model weights.

---

### Five Core Subsystems

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                               META-EVOLVER SCAFFOLDING                                │
├───────────────────────────────┬───────────────────────────────┬───────────────────────┤
│ 1. ScaffoldAssert             │ 2. GEPAPromptOptimizer        │ 3. ScaffoldRLM        │
│    • Intra-step verification  │    • Modular prompt AST       │    • Variable vs Token│
│    • Hard Assert vs Suggest   │    • Multi-task Pareto search │    • Sandboxed REPL   │
│    • Zero step-penalty retry  │    • Crossover merging        │    • Sub-LLM queries  │
├───────────────────────────────┴───────────────────────────────┼───────────────────────┤
│ 4. FlexScaffold                                               │ 5. TelemetryTracer    │
│    • Evolving Python module source code                       │    • Step-level spans │
│    • Sandboxed AST compilation & exec                         │    • MLflow 2.16+ UI  │
│    • Crash-resilient FlexRule harness wrappers                │    • OpenTelemetry    │
└───────────────────────────────────────────────────────────────┴───────────────────────┘
```

1. **`ScaffoldAssert` (`tools/assertions.py`)**:
   - `NonEmptyArgsAssertion`, `ValidToolAssertion`, `AdmissibleCommandAssertion`, `NumericRangeAssertion`.
   - Intercepts invalid parameters before environment execution $\to$ routes to `assert_retry` node with actionable feedback $\to$ saves 100% of wasted environment steps.
2. **`GEPAPromptOptimizer` (`prompts/gepa.py`)**:
   - Decomposes system instruction into `ModularPrompt` components: `planning`, `tool_policy`, `error_recovery`, `core_role`.
   - Populates a multi-task Pareto candidate frontier; mutates via reflection on failure traces and merges non-dominated parents.
3. **`ScaffoldRLM` (`tools/repl.py`)**:
   - Resolves context rot by storing massive raw observations ($>4\text{KB}$) in an isolated REPL session.
   - Prompt receives lightweight `VariableDescriptor` metadata; agent executes Python slices and delegates semantic reads via bounded `llm_query` calls ($>70\%$ token reduction).
4. **`FlexScaffold` (`core/flex.py`)**:
   - Treats executable Python module code as an optimizable parameter (`FlexModule`).
   - Dynamic tools and `FlexRule` harness wrappers run in sandboxes; runtime exceptions are caught as diagnostic reflection feedback rather than crashing the run.
5. **`TelemetryTracer` (`telemetry/tracer.py`)**:
   - Hierarchical span tracking with node-level latencies, tokens, and tool payloads.
   - Dual export to **MLflow Tracing format** and **OpenTelemetry JSON**.

---

### Visual Style Directive for Gemini Image Generation
* **Aesthetic**: Technical flowchart and modular subsystem architecture diagram on pure white background.
* **Layout**: Clean grid with color-coded cards and connection flowcharts mapping from the agent thinking step to the five declarative compiler subsystems.
