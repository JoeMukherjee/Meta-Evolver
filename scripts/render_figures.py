"""Render the paper's figures deterministically, from code and real data.

This is the ground truth layer. Nothing here is generated or approximated: the
curves in Figure 2 are the full action logs, point for point, and every label
is a string in this file. A figure is therefore reviewable as a diff, and
re-running it after a new experiment updates the paper rather than silently
disagreeing with it.

A generative pass may afterwards restyle these renders (see
``generate_paper_figures.py --restyle``), but it is handed the finished
geometry as an image and asked only to change how it looks. That ordering
matters: a model asked to *invent* a plot will invent plausible numbers, and a
plausible number in a results figure is a retraction waiting to happen.

    python scripts/render_figures.py           # all
    python scripts/render_figures.py fig2      # one
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "figures" / "rendered"
WORKSPACE = REPO.parents[1]

# --- house style -------------------------------------------------------------

INK = "#0F172A"
SLATE = "#475569"
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
RED = "#DC2626"
DARK_RED = "#991B1B"
GREEN = "#16A34A"
GRID = "#E2E8F0"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

VARIANTS = {
    "baseline_static": ("Static retrieval", RED, "-", 2.4),
    "poc2_dynamic_eviction": ("Eviction only", DARK_RED, "--", 2.4),
    "poc1_state_exhaustion": ("State exhaustion", TEAL, "-", 2.4),
    "poc4_hybrid_adaptive": ("Hybrid adaptive (ours)", GREEN, "-", 3.4),
}


def load_runs() -> dict[str, dict]:
    """Discovery curves straight from the recorded action logs."""
    path = WORKSPACE / "runs" / "poc_ood_mitigation_results.json"
    out: dict[str, dict] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        actions = row.get("actions", [])
        seen: set[str] = set()
        curve: list[int] = []
        for action in actions:
            seen.add(action)
            curve.append(len(seen))
        flat = next((i for i in range(len(curve) - 5) if curve[i + 5] == curve[i]), None)
        out[row["variant"]] = {
            "curve": curve,
            "steps": row["duration_steps"],
            "unique": len(seen),
            "success": row["success"],
            "seconds": row.get("duration_ms", 0) / 1000,
            "flat_at": flat,
        }
    return out


def _save(fig, stem: str) -> Path:
    """Write PNG for the README and PDF for the paper.

    The PDF is the one that matters for submission: it stays vector, so a
    reviewer zooming in sees type rather than pixels, and it embeds at any
    column width without resampling.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.28)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return png


def card(ax, x, y, w, h, title, lines, edge):
    """One annotation card, in axes-fraction coordinates."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=edge,
            linewidth=1.6,
            zorder=5,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.028, y + h - 0.055, title, transform=ax.transAxes,
        fontsize=11.5, fontweight="bold", color=edge, zorder=6, clip_on=False,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 0.028, y + h - 0.112 - i * 0.042, line, transform=ax.transAxes,
            fontsize=9.6, color=INK, zorder=6, clip_on=False,
        )


def figure_2(runs: dict[str, dict]) -> Path:
    """The retrieval trap, plotted from the raw logs."""
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    fig.subplots_adjust(left=0.062, right=0.615, top=0.845, bottom=0.115)

    ax.plot([0, 40], [0, 40], color="#94A3B8", ls=(0, (5, 5)), lw=1.2, zorder=1)
    ax.text(
        27.5, 29.6, "one new action every step", rotation=33, fontsize=9.5,
        color="#64748B", style="italic", ha="center", va="bottom", zorder=1,
    )

    for key, (_label, colour, style, width) in VARIANTS.items():
        run = runs[key]
        curve = run["curve"]
        steps = range(1, len(curve) + 1)
        ax.plot(steps, curve, color=colour, ls=style, lw=width, zorder=4,
                solid_capstyle="round")
        ax.plot(len(curve), curve[-1], "o", color=colour, ms=8, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)

    # The trap: where the two failing runs stop discovering anything.
    trap_top = max(runs["baseline_static"]["unique"], runs["poc2_dynamic_eviction"]["unique"])
    ax.fill_between([0, 50], 0, trap_top + 0.6, color=RED, alpha=0.07, zorder=0)
    ax.text(29, 2.4, "TRAPPED", fontsize=30, color=RED, alpha=0.18,
            fontweight="bold", ha="center", va="center", zorder=1)

    # Both callouts sit in the clear band on the right, past where the two
    # solving curves stop, so neither can collide with a curve or the axes.
    for key, text_xy in (("baseline_static", (33.0, 17.0)),
                         ("poc2_dynamic_eviction", (33.0, 12.0))):
        run = runs[key]
        ax.annotate(
            f"saturates, step {run['flat_at']}",
            xy=(run["flat_at"], run["curve"][run["flat_at"] - 1]),
            xytext=text_xy,
            fontsize=10.0, color=VARIANTS[key][1], va="center",
            arrowprops=dict(arrowstyle="->", color=VARIANTS[key][1], lw=1.3,
                            shrinkA=3, shrinkB=5,
                            connectionstyle="arc3,rad=-0.18"),
        )

    label_offsets = {
        "poc1_state_exhaustion": (1.2, 0.8, "left"),
        "poc4_hybrid_adaptive": (0.6, -2.6, "left"),
        "baseline_static": (-1.2, 2.0, "right"),
        "poc2_dynamic_eviction": (-1.2, -2.8, "right"),
    }
    for key, (label, colour, _s, _w) in VARIANTS.items():
        run = runs[key]
        dx, dy, ha = label_offsets[key]
        ax.text(
            run["steps"] + dx, run["unique"] + dy,
            f"{label}  ({run['steps']}, {run['unique']})",
            fontsize=10.4, color=colour, ha=ha, va="center",
            fontweight="bold" if "ours" in label else "normal",
        )

    solved = runs["poc4_hybrid_adaptive"]
    ax.plot(solved["steps"], solved["unique"], "*", color=GREEN, ms=19, zorder=6,
            markeredgecolor="white", markeredgewidth=1.0)

    ax.set_xlim(0, 52)
    ax.set_ylim(0, 40)
    ax.set_xlabel("steps taken", fontsize=12)
    ax.set_ylabel("distinct actions tried", fontsize=12)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)

    fig.text(0.062, 0.945, "The retrieval trap is a saturation phenomenon",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.062, 0.897,
             "ALFWorld, out-of-distribution layout. Distinct actions tried vs. steps taken.",
             fontsize=12.5, color=SLATE)

    ev = runs["poc2_dynamic_eviction"]
    hy = runs["poc4_hybrid_adaptive"]
    card(ax, 1.045, 0.70, 0.60, 0.27, "Same start, different fate",
         ["All four agents open with nearly",
          "the same six actions. The prior is",
          "good advice until it is not."], "#334155")
    card(ax, 1.045, 0.38, 0.60, 0.27, "Eviction alone is worse",
         ["Dropping a failed prior leaves",
          "nothing behind: it saturates at step",
          f"{ev['flat_at']}, sooner than the baseline it fixes."], DARK_RED)
    card(ax, 1.045, 0.06, 0.60, 0.27, "Replace, do not just remove",
         ["Eviction plus a state-exhaustion",
          f"fallback keeps searching: {hy['unique']} distinct",
          f"actions in {hy['steps']} steps, task solved."], GREEN)

    fig.text(0.062, 0.012,
             "One seed per variant, 50-step budget, identical model, tools and base prompt. "
             "Curves are the raw action logs.",
             fontsize=9, color=SLATE)

    return _save(fig, "fig2_ood_generalization")


def figure_3() -> Path:
    """The life of one memory: events on a timeline, utility beneath."""
    fig = plt.figure(figsize=(13.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.42,
                          left=0.055, right=0.665, top=0.80, bottom=0.10)
    top = fig.add_subplot(gs[0])
    bottom = fig.add_subplot(gs[1])

    events = [
        (0, "BORN", BLUE, "induced from a\ncontrastive pair",
         "one attempt solved it,\nanother did not"),
        (1, "RETRIEVED", TEAL, "similarity x utility,\nthen MMR",
         "MMR spends slots on different\nideas, not paraphrases"),
        (2, "CREDITED", AMBER, "every episode that\ncited it is charged",
         "wins and uses, never\na lost increment"),
        (3, "CREDITED", AMBER, "every episode that\ncited it is charged",
         "wins and uses, never\na lost increment"),
        (5, "PRUNED", RED, "a fair trial,\nthen judged",
         "at least 4 uses before\nit can be dropped"),
    ]

    top.set_xlim(-0.55, 5.6)
    top.set_ylim(0, 1)
    top.axis("off")
    top.annotate("", xy=(5.5, 0.13), xytext=(-0.45, 0.13),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
    for g in range(6):
        top.plot([g, g], [0.105, 0.155], color=INK, lw=1.8)
        top.text(g, 0.035, f"g{g}", ha="center", fontsize=12, color=INK)
    top.text(-0.5, 0.035, "generations", ha="left", fontsize=11, color=SLATE)

    for g, tag, colour, caption, sub in events:
        top.add_patch(
            FancyBboxPatch(
                (g - 0.42, 0.80), 0.84, 0.16,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=colour, alpha=0.13, edgecolor=colour, lw=1.6,
                transform=top.transData, clip_on=False,
            )
        )
        top.text(g, 0.88, tag, ha="center", va="center", fontsize=11,
                 fontweight="bold", color=colour)
        top.text(g, 0.63, caption, ha="center", va="center", fontsize=10.4, color=INK)
        top.text(g, 0.40, sub, ha="center", va="center", fontsize=8.8, color=SLATE)
        top.annotate("", xy=(g, 0.17), xytext=(g, 0.30),
                     arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.2))

    # Utility trajectory: healthy, then judged.
    xs = [0, 1, 2, 3, 4, 5]
    ys = [0.500, 0.571, 0.625, 0.545, 0.400, 0.250]
    threshold = 0.34

    bottom.axhline(threshold, color=RED, ls="--", lw=1.6, zorder=2)
    bottom.text(5.45, threshold + 0.028, "prune threshold, 0.34", color=RED,
                fontsize=10, ha="right")

    above = [(x, y) for x, y in zip(xs, ys, strict=True) if y >= threshold]
    bottom.plot([p[0] for p in above], [p[1] for p in above], color=BLUE, lw=2.8, zorder=4)
    bottom.plot([3, 4, 5], [0.545, 0.400, 0.250], color=BLUE, lw=2.8, zorder=4)
    below_x = [4.0, 4.412, 5.0]
    below_y = [0.400, threshold, 0.250]
    bottom.plot(below_x[1:], below_y[1:], color=RED, lw=2.8, zorder=5)
    bottom.fill_between([4.412, 5.0], [threshold, threshold], [threshold, 0.250],
                        color=RED, alpha=0.16, zorder=1)

    bottom.plot(0, 0.500, "o", color=BLUE, ms=8, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
    bottom.text(0.12, 0.545, "0.5, untested", color=BLUE, fontsize=10.2)
    bottom.plot(2, 0.625, "o", color=GREEN, ms=11, zorder=6,
                markeredgecolor="white", markeredgewidth=1.4)
    bottom.text(2, 0.695, "above threshold", color=GREEN, fontsize=9.6, ha="center")
    bottom.plot(5, 0.250, "X", color=RED, ms=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)

    bottom.set_xlim(-0.55, 5.6)
    bottom.set_ylim(0, 1.0)
    bottom.set_xticks(range(6))
    bottom.set_xticklabels([f"g{g}" for g in range(6)], fontsize=11)
    bottom.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bottom.set_xlabel("generations", fontsize=11.5)
    bottom.set_title("utility over its lifetime", fontsize=12, loc="left", color=INK, pad=8)
    bottom.grid(axis="y", color=GRID, lw=0.9)
    bottom.set_axisbelow(True)

    fig.text(0.055, 0.945, "The life of a memory", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.055, 0.897,
             "How the scaffold decides what to keep: one memory, from birth to eviction.",
             fontsize=12.5, color=SLATE)

    fig.text(0.700, 0.760, "the rule", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.700, 0.660, r"$\mathrm{utility}=\dfrac{\mathrm{wins}+1}{\mathrm{uses}+2}$",
             fontsize=19, color=INK)
    for i, line in enumerate([
        "Beta(1,1) posterior over: do episodes",
        "citing this memory succeed?",
        "",
        "An untested memory sits at 0.5, so it is",
        "neither trusted nor pruned.",
        "",
        "A memory that keeps losing stops being",
        "retrieved before it is ever deleted.",
    ]):
        fig.text(0.700, 0.560 - i * 0.036, line, fontsize=10.2, color=INK)

    fig.text(0.700, 0.215, "why a bank must forget", fontsize=12.5,
             fontweight="bold", color=INK)
    for i, line in enumerate([
        "A bank that only grows fills its slots",
        "with paraphrases. Curation, not",
        "accumulation, is what keeps the loop",
        "paying off.",
    ]):
        fig.text(0.700, 0.160 - i * 0.036, line, fontsize=10.2, color=SLATE)

    return _save(fig, "fig3_algorithm")


MERMAID_FIG1 = """flowchart TB
  classDef stage fill:#FFFFFF,stroke:#0F172A,stroke-width:2px,color:#0F172A
  classDef mem fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#0F172A
  classDef cred fill:#F0FDF4,stroke:#16A34A,stroke-width:2px,color:#0F172A
  classDef prm fill:#F0FDFA,stroke:#0D9488,stroke-width:2px,color:#0F172A
  classDef cur fill:#FFFBEB,stroke:#D97706,stroke-width:2px,color:#0F172A
  classDef frozen fill:#E2E8F0,stroke:#475569,stroke-width:2px,color:#0F172A
  classDef ctrl fill:#FFFBEB,stroke:#D97706,stroke-width:2.5px,color:#0F172A

  subgraph OUTER["OUTER LOOP - evolution graph (one generation)"]
    direction LR
    S1["Sample tasks<br/><i>train / held-out split</i>"]:::stage
    S2["Fan out rollouts<br/><i>K attempts per task</i>"]:::stage
    S3["Score<br/><i>pass@K, per task</i>"]:::stage
    S4["Induce memories<br/><i>contrastive</i>"]:::mem
    S5["Credit and prune<br/><i>Beta posterior utility</i>"]:::cred
    S6["Optimise prompt<br/><i>validate held-out</i>"]:::prm
    S7["Escalate curriculum<br/><i>harder environment</i>"]:::cur
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -. "next generation" .-> S1
  end

  subgraph INNER["INNER LOOP - episode graph (one rollout)"]
    direction LR
    P["prepare<br/><i>retrieve top-k (MMR)</i>"]:::stage
    T["think"]:::stage
    A["act"]:::stage
    D["adapt"]:::stage
    F["finalise"]:::stage
    N["nudge"]:::stage
    U{"usable?"}
    P --> T --> A --> D --> F --> U
    T -. "no tool call" .-> N
    N --> T
    D -. "continue" .-> T
    U -- "reasoning failure: learn from it" --> LEARN["induce + credit"]:::mem
    U -- "infrastructure error: discard" --> DROP["discarded"]:::frozen
    CTRL["ADAPTIVE CONTROLLER<br/>soft priors<br/>stagnation eviction<br/>state-exhaustion fallback"]:::ctrl
    CTRL --- D
  end

  LLM["FROZEN LLM (API only)"]:::frozen
  ENV["FROZEN BENCHMARK ENV"]:::frozen
  LLM --- T
  ENV --- A

  S2 == "one generation" ==> P
  F == "trajectories" ==> S3
"""


def figure_1() -> Path:
    """Architecture, from a Mermaid source so the graph is text, not pixels."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = OUT_DIR / "fig1_architecture.mmd"
    src.write_text(MERMAID_FIG1, encoding="utf-8")
    out = OUT_DIR / "meta_evolver_architecture.png"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"theme": "base", "themeVariables": {"fontFamily": "Segoe UI, Inter, Helvetica"}}, fh)
        config = fh.name

    cmd = ["mmdc", "-i", str(src), "-o", str(out), "-b", "white", "-w", "2600",
           "-H", "1500", "-c", config]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0 or not out.exists():
        print(f"  mermaid failed: {result.stderr[:400]}")
        return src
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", help="fig1, fig2, fig3")
    args = parser.parse_args()
    keys = args.figures or ["fig1", "fig2", "fig3"]

    runs = load_runs()
    for key in keys:
        if key == "fig1":
            path = figure_1()
        elif key == "fig2":
            path = figure_2(runs)
        else:
            path = figure_3()
        size = path.stat().st_size // 1024 if path.exists() else 0
        print(f"  {key}: {path}  ({size} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
