"""Two-minute quickstart: watch an agent get better over four generations.

    python examples/quickstart.py

Needs GEMINI_API_KEY (or --model and the matching key for another provider).
Costs a few hundred model calls. Run `examples/offline_demo.py` first if you
want to see the machinery work without spending anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_evolver import EvolutionConfig, MetaEvolver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="devops")
    parser.add_argument("--model", default="gemini/gemini-3-flash")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--memory", default="memories.jsonl")
    args = parser.parse_args()

    evolver = MetaEvolver(
        benchmark=args.benchmark,
        model=args.model,
        memory_path=args.memory,
        config=EvolutionConfig(generations=args.generations, max_steps=15),
    )

    print(f"Evolving on {evolver.benchmark.name} with {evolver.client.model}\n")

    # Measure before learning anything, so the end-of-run number has a baseline
    # to be compared against. A curve that starts at generation 1 is not a curve.
    baseline = evolver.evaluate(split="eval", use_memory=False)
    print(f"baseline (held-out, no memory): {baseline['pass_rate'] * 100:.1f}%\n")

    evolver.evolve(on_report=lambda report: print(report.render()))

    print("\n" + evolver.render_progress())

    final = evolver.evaluate(split="eval")
    print(
        f"\nheld-out after evolution: {final['pass_rate'] * 100:.1f}% "
        f"(baseline {baseline['pass_rate'] * 100:.1f}%)"
    )

    saved = evolver.save()
    print(f"\nlearned memory -> {saved['memory']}")
    print(f"evolved prompt -> {saved['prompt']}")
    for memory in sorted(evolver.bank, key=lambda m: -m.utility)[:3]:
        print(f"  [{memory.utility:.2f} over {memory.uses} uses] {memory.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
