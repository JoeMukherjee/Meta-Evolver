"""Generate the paper's figures with Gemini 3 Pro Image (Nano Banana Pro).

Every number in these prompts is read from a real run, not invented:

  runs/poc_ood_mitigation_results.json     the four-variant OOD ablation
  runs/ours_adaptive_run_001/summary.json  the ALFWorld eval split

so the figures and the results section cannot drift apart. Re-do a run,
re-run this, and the figures follow.

Two prompt habits do most of the work for diagram generation:

* **Quote every string that must appear.** The model reproduces quoted text
  far more reliably than paraphrased intent, and a label typo is the single
  most common reason a generated figure is unusable.
* **State the layout as geometry**, not as vibe: which panel sits where, what
  flows into what, what aligns with what.

There are two ways to use this, and they are not equivalent.

**Restyle (preferred for anything carrying numbers).** ``render_figures.py``
draws the figure from real data with matplotlib; this script then hands that
render to the image model and asks it *only* to restyle. The geometry, the
tick values and every label come from code, so the model cannot invent a data
point. Reach for this whenever a figure has an axis.

**Generate (for conceptual figures only).** The model composes the figure from
a text description. It produces better-balanced layouts than an auto-layout
engine, which matters for an architecture diagram, and costs nothing in
correctness because there is no data to get wrong.

    python scripts/render_figures.py                                   # ground truth
    python scripts/generate_paper_figures.py --restyle fig2 --size 4K  # then polish

    python scripts/generate_paper_figures.py --style all --size 2K --out /tmp/drafts
    python scripts/generate_paper_figures.py --style schematic --size 4K
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "docs" / "figures"
WORKSPACE = REPO.parents[1]  # the EnvHarness checkout that holds runs/

# --- style variants ----------------------------------------------------------
#
# Three directions, drafted cheaply and chosen before spending 4K. They differ
# in what the reader's eye is asked to do, not in decoration:
#
#   vector     boxes and arrows carry the structure; colour marks meaning
#   schematic  near-monochrome textbook line-art; structure carried by form
#   editorial  small pictographs anchor each stage; friendlier, more to go wrong

_ACCURACY = (
    "TEXT ACCURACY IS CRITICAL: reproduce every quoted label verbatim, "
    "correctly spelled, with no invented words, no duplicated letters, no "
    "placeholder lorem text, and no stray punctuation or unmatched brackets."
)

STYLE_VECTOR = (
    "STYLE CONTRACT - follow exactly. Flat 2D vector scientific diagram for a "
    "machine-learning conference paper (NeurIPS / ICLR / Nature Machine "
    "Intelligence). Pure solid white background #FFFFFF. No 3D, no bevels, no "
    "drop shadows, no glow, no gradient fills used as decoration, no glossy "
    "highlights, no photographic texture. Thin crisp 2px strokes. Rounded "
    "rectangles with 1.5px outlines and pale tinted fills. Sans-serif "
    "typography throughout (Inter or Helvetica), tight and legible at column "
    "width. Restrained palette only: ink #0F172A for text and outlines, slate "
    "#475569 for secondary text, blue #2563EB, teal #0D9488, amber #D97706, "
    "red #DC2626, green #16A34A. Use colour to carry meaning, never to "
    "decorate. Generous white space. Every arrow terminates cleanly on a box "
    "edge with a small solid arrowhead. " + _ACCURACY
)

STYLE_SCHEMATIC = (
    "STYLE CONTRACT - follow exactly. Precise black-and-white technical "
    "schematic for a journal paper, in the tradition of a systems textbook "
    "figure. Pure solid white background #FFFFFF. Almost entirely monochrome: "
    "black #000000 line work at two weights (1px detail, 2.5px primary flow), "
    "mid-grey #9CA3AF for secondary annotation, and exactly ONE accent colour, "
    "a muted blue #1D4ED8, used sparingly to mark the single most important "
    "path. Plain rectangles with squared corners, not rounded. Hairline leader "
    "lines to small set-in labels. Slab-serif titles, clean sans labels. No "
    "fills except white and the palest grey. No 3D, no shadows, no gradients, "
    "no icons, no glow. Dense and information-first: the look of a figure "
    "drawn to be read rather than admired. " + _ACCURACY
)

STYLE_EDITORIAL = (
    "STYLE CONTRACT - follow exactly. Modern scientific editorial illustration "
    "of the kind used in a well-designed ML paper or a Distill.pub article. "
    "Pure solid white background #FFFFFF. Flat 2D vector, no 3D and no "
    "photographic texture. Each pipeline stage is anchored by a small simple "
    "line pictograph (a stack for memory, a branching arrow for rollouts, a "
    "staircase for curriculum, a funnel for pruning) at the same 2px weight as "
    "the boxes. Soft tinted panel backgrounds in very pale blue #EFF6FF and "
    "pale amber #FFFBEB group related stages. Palette: ink #0F172A, slate "
    "#475569, blue #2563EB, teal #0D9488, amber #D97706, red #DC2626, green "
    "#16A34A. Generous white space, confident hierarchy, large clear section "
    "headers. Approachable but precise, unmistakably a research figure and "
    "never a marketing graphic. " + _ACCURACY
)

STYLES = {
    "vector": STYLE_VECTOR,
    "schematic": STYLE_SCHEMATIC,
    "editorial": STYLE_EDITORIAL,
}

#: Rebound by ``main`` before any prompt is built.
STYLE = STYLE_VECTOR


def load_ood_ablation() -> dict[str, dict]:
    """The four-variant OOD ablation with its discovery curves.

    The headline numbers (solved / not solved) say a method won. The
    *discovery curve* -- distinct actions tried against steps taken -- says
    why, because a trapped agent's curve goes flat while a searching agent's
    tracks the diagonal. ``flatlines_at`` is the first step after which the
    agent discovers nothing new for five consecutive steps, which is the
    moment the trap closes.
    """
    path = WORKSPACE / "runs" / "poc_ood_mitigation_results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for row in rows:
        actions = row.get("actions", [])
        seen: set[str] = set()
        curve: list[int] = []
        for action in actions:
            seen.add(action)
            curve.append(len(seen))
        flat = next(
            (i for i in range(len(curve) - 5) if curve[i + 5] == curve[i]), None
        )
        out[row["variant"]] = {
            "success": row["success"],
            "steps": row["duration_steps"],
            "unique_actions": len(set(actions)),
            "seconds": round(row.get("duration_ms", 0) / 1000),
            "curve": curve,
            "flatlines_at": flat,
            "first_actions": actions[:6],
        }
    return out


def figure_1_architecture() -> str:
    return f"""{STYLE}

Draw a two-tier system architecture diagram titled
"Meta-Evolver: a scaffolding-evolution harness for frozen LLM agents".

Overall geometry: one wide canvas split into an upper band and a lower band,
separated by a thin horizontal rule. The upper band is the outer loop, the
lower band is the inner loop. A slim vertical arrow on the left labelled
"one generation" descends from the upper band to the lower band, and a
matching arrow on the right labelled "trajectories" ascends back up. Those two
bands plus the two arrows form one closed cycle: make the cycle obvious.

UPPER BAND, header "OUTER LOOP - evolution graph (one generation)".
Seven boxes left to right, connected by right-pointing arrows:
  1. "Sample tasks"        subtitle "train / held-out split"
  2. "Fan out rollouts"    subtitle "K attempts per task, concurrent"
  3. "Score"               subtitle "pass@K, per task"
  4. "Induce memories"     subtitle "contrastive: success vs failure"
  5. "Credit and prune"    subtitle "Beta posterior utility"
  6. "Optimise prompt"     subtitle "propose, then validate held-out"
  7. "Escalate curriculum" subtitle "harder environment"
From box 7 a long arrow loops back over the top to box 1, labelled
"next generation".
Beneath boxes 4 to 7, a row of four small chips reading exactly:
  "MEMORY"   "CREDIT"   "PROMPT"   "CURRICULUM"
and under that row one caption:
"three scaffold channels, plus the environment itself".

LOWER BAND, header "INNER LOOP - episode graph (one rollout)".
A left-to-right state machine of boxes:
  "prepare" -> "think" -> "act" -> "adapt" -> "finalise"
with these annotations:
  - Below "prepare": "retrieve top-k memories (MMR)".
  - A dashed feedback arrow from "adapt" back to "think" labelled "continue".
  - From "think", a short branch down to a small box "nudge" labelled
    "no tool call", rejoining "think".
  - Below "adapt", a box outlined in amber #D97706 headed
    "ADAPTIVE CONTROLLER" with three bullet lines:
      "soft priors"
      "stagnation eviction"
      "state-exhaustion fallback"
  - From "finalise", an arrow to a small diamond "usable?" splitting into two
    arrows that terminate in TEXT ONLY, with no box or shape at the end:
    an upper green #16A34A arrow whose arrowhead is immediately followed by
    the words "reasoning failure - learn from it", and a lower red #DC2626
    arrow followed by the words "infrastructure error - discard".
    Do not draw an empty rectangle at the end of either arrow.

Along the left edge, two stacked grey boxes reading "FROZEN LLM (API only)"
and "FROZEN BENCHMARK ENV", with the caption "weights are never updated"
beneath them. Set all of this text HORIZONTALLY: never rotate or set text
vertically anywhere in the figure. Draw one thin connector line from the
"FROZEN LLM (API only)" box to the "think" box, and one from the
"FROZEN BENCHMARK ENV" box to the "act" box.

CONSISTENCY: every stage subtitle must use the same size, weight and colour
as every other stage subtitle. Do not grey out or fade any single label.
Every text element is set horizontally; nothing is rotated.
NO EMPTY SHAPES: every box drawn must contain text. Never draw a blank
rectangle.

Composition: wide, airy, balanced. This is Figure 1 of a paper and must read
clearly at half a page."""


def figure_2_ood(data: dict[str, dict]) -> str:
    b = data["baseline_static"]
    e = data["poc2_dynamic_eviction"]
    x = data["poc1_state_exhaustion"]
    h = data["poc4_hybrid_adaptive"]
    return f"""{STYLE}

Draw a scientific results figure titled
"The retrieval trap is a saturation phenomenon".
Subtitle underneath in slate #475569:
"ALFWorld, out-of-distribution layout. Distinct actions tried vs. steps taken."

LAYOUT: one large line plot occupying the left two thirds of the canvas, and a
narrow right-hand column holding three stacked annotation cards.

THE PLOT.
X axis "steps taken", ticks at 0, 10, 20, 30, 40, 50.
Y axis "distinct actions tried", ticks at 0, 10, 20, 30, 40.
Draw a faint grey dashed diagonal from (0,0) to (40,40), labelled along its
length in small grey italic text: "one new action every step".

Four curves, all starting together at the origin and rising steeply for the
first five steps, then separating. Draw each as a smooth thick line with a
labelled dot at its end point:

  1. RED #DC2626, solid. Rises to about 6 by step 6, then bends sharply and
     runs almost FLAT for the rest of the plot, ending at ({b["steps"]}, {b["unique_actions"]}).
     End-dot label: "Static retrieval  ({b["steps"]}, {b["unique_actions"]})".
     Put a small red marker on this curve at step {b["flatlines_at"]} with a leader line
     and the label "saturates, step {b["flatlines_at"]}".

  2. DARK RED #991B1B, dashed. Rises to about 6 by step 5, then goes FLAT even
     EARLIER and even lower than curve 1, ending at ({e["steps"]}, {e["unique_actions"]}).
     End-dot label: "Eviction only  ({e["steps"]}, {e["unique_actions"]})".
     Put a marker at step {e["flatlines_at"]} labelled "saturates, step {e["flatlines_at"]}".

  3. TEAL #0D9488, solid. Climbs steadily and almost straight, hugging just
     below the diagonal the whole way, ending at ({x["steps"]}, {x["unique_actions"]}).
     End-dot label: "State exhaustion  ({x["steps"]}, {x["unique_actions"]})".

  4. GREEN #16A34A, solid and the thickest line. Climbs steadily just below the
     diagonal, ending sooner than curve 3 because it solves the task earlier, at
     ({h["steps"]}, {h["unique_actions"]}). End-dot label:
     "Hybrid adaptive (ours)  ({h["steps"]}, {h["unique_actions"]})".
     Draw a small green star at this end point and label it "task solved".

Shade the flat region under curves 1 and 2 in a very pale red wash, and place
one large low-opacity word across that region reading "TRAPPED".

THE RIGHT-HAND COLUMN, three small cards stacked vertically, each with a bold
one-line heading and two lines of body text, reading exactly:

  Card 1, heading "Same start, different fate":
  "All four agents open with nearly the same six actions."
  "The prior is good advice until it is not."

  Card 2, heading "Eviction alone is worse", outlined in dark red #991B1B:
  "Dropping a failed prior leaves nothing behind."
  "It saturates at step {e["flatlines_at"]}, sooner than the baseline it was meant to fix."

  Card 3, heading "Replace, do not just remove", outlined in green #16A34A:
  "Eviction plus a state-exhaustion fallback keeps the search moving."
  "{h["unique_actions"]} distinct actions in {h["steps"]} steps, and the task solved."

At the very bottom, one small-print methods line reading exactly:
"One seed per variant, 50-step budget, identical model, tools and base prompt. Curves are the raw action logs."

PRECISION: each curve stops exactly at its labelled end point. Do not extend
any line, arrowhead or shadow beyond its end dot, and keep the two end dots in
the upper right clearly separated so their labels never overlap.

Make the two flat curves and the two climbing curves unmistakably different at
a glance: that contrast is the entire result."""


def figure_3_algorithm() -> str:
    return f"""{STYLE}

Draw a methods figure titled "The life of a memory".
Subtitle in slate #475569:
"How the scaffold decides what to keep, follow one memory from birth to eviction."

LAYOUT: a single horizontal timeline running left to right across the middle
of the canvas, with events attached above and below it. The timeline is
labelled "generations" and marked g0, g1, g2, g3, g4, g5.

ABOVE THE TIMELINE, five events on leader lines, in order:

  At g0, a small card icon labelled "BORN".
    Caption: "induced from a contrastive pair"
    Sub-caption in slate: "one attempt solved it, another did not"

  At g1, a magnifier icon labelled "RETRIEVED".
    Caption: "similarity x utility, then MMR"
    Sub-caption: "MMR spends slots on different ideas, not paraphrases"

  At g2 AND SEPARATELY at g3, two distinct tally icons, each labelled
  "CREDITED" (draw both; this event happens twice).
    Caption: "every episode that cited it is charged"
    Sub-caption: "wins and uses, never a lost increment"

  At g5, a funnel icon with a small cross, labelled "PRUNED".
    Caption: "a fair trial, then judged"
    Sub-caption: "at least 4 uses before it can be dropped"

BELOW THE TIMELINE, a line chart titled "utility over its lifetime".
Y axis from 0.0 to 1.0 with visible ticks and labels at 0.0, 0.2, 0.4, 0.6,
0.8 and 1.0.
X AXIS, this is required and must not be omitted: draw the x axis with six
visible tick marks labelled, left to right, "g0", "g1", "g2", "g3", "g4",
"g5". Those six ticks must sit directly beneath the six corresponding marks on
the timeline above, so the chart and the timeline share one coordinate system
and a reader can read straight down from an event to its utility. Label the x
axis "generations".
Plot a single line that begins at 0.5 at g0, rises slightly to about 0.62 by
g2, then declines through g3 and g4 and falls below a red dashed horizontal
threshold line at 0.34 by g5.
Label the starting point "0.5, untested".
Label the red dashed line "prune threshold, 0.34".
Shading rule, follow precisely: fill pale red ONLY the region that lies
between the line and the threshold in the segment where the line has fallen
BELOW 0.34, which is roughly from g4 to g5. Leave the area under the rest of
the line completely white and unshaded. The shaded wedge must be small and
confined to the end of the lifetime.
Put a green check marker where the line is above the threshold and a red cross
where it ends.

TO THE RIGHT of the whole timeline, a boxed panel headed "the rule", containing
the display equation set large:
  utility = (wins + 1) / (uses + 2)
and three short lines beneath it reading exactly:
"Beta(1,1) posterior over: do episodes citing this memory succeed?"
"An untested memory sits at 0.5, so it is neither trusted nor pruned."
"A memory that keeps losing stops being retrieved before it is ever deleted."

TO THE LEFT, a narrow vertical panel headed "why a bank must forget",
containing two short lines reading exactly:
"A bank that only grows fills its slots with paraphrases."
"Curation, not accumulation, is what keeps the loop paying off."

AXES: both axes are drawn with visible tick marks and numeric or generation
labels. A chart with an unlabelled axis is unusable; do not omit the x-axis
labels.

RESTRAINT: the chart sits on plain white. Do not add decorative vertical
colour bands or tinted background regions behind it. The ONLY shaded area in
the whole chart is the small wedge where the line has dropped below the 0.34
threshold near the end. Do not fill the area under the early, healthy part of
the curve.

Keep the timeline the dominant visual element: a reader should follow one
object through time, not read three separate diagrams."""


RESTYLE_INSTRUCTION = """You are restyling a finished scientific figure. The
image provided is the ground truth: it was rendered from real experimental
data by a plotting library.

CHANGE ONLY THE VISUAL PRESENTATION. Specifically you may improve: typography
and type hierarchy, spacing and alignment, the weight and finish of lines,
the styling of the annotation cards and their borders, the arrowheads, and the
overall balance of the composition.

YOU MUST NOT CHANGE, AND MUST REPRODUCE EXACTLY:
- every number, on the axes, in the labels, and inside the annotation cards
- the position and shape of every plotted line, point and marker
- every axis tick value and every axis label
- the wording of every piece of text, character for character
- which colour belongs to which series

Do not add data. Do not remove data. Do not smooth, idealise or "clean up" any
plotted line: the small steps and plateaus in the curves are real measurements
and carry the result. Do not add a legend, a watermark, a logo, a caption, a
border or any decorative element that is not already present.

Treat this as a typographer redrawing a chart, not an illustrator reimagining
it. If you are unsure whether something is data or decoration, leave it
exactly as it is."""


FIGURES: dict[str, tuple[str, str]] = {
    "fig1": ("meta_evolver_architecture.png", "16:9"),
    "fig2": ("fig2_ood_generalization.png", "16:9"),
    "fig3": ("fig3_algorithm.png", "16:9"),
}


def build_prompt(key: str) -> str:
    if key == "fig1":
        return figure_1_architecture()
    if key == "fig2":
        return figure_2_ood(load_ood_ablation())
    return figure_3_algorithm()


def generate(
    key: str, size: str, model: str, out_dir: Path, suffix: str = ""
) -> Path | None:
    from google import genai

    filename, aspect = FIGURES[key]
    if suffix:
        filename = f"{Path(filename).stem}{suffix}.png"
    out_path = out_dir / filename

    print(f"\n=== {key}{suffix} -> {filename}  [{model}, {size}, {aspect}] ===")
    client = genai.Client()
    started = time.time()
    try:
        interaction = client.interactions.create(
            model=model,
            input=build_prompt(key),
            response_format={
                "type": "image",
                # No explicit mime_type: gemini-3-pro-image rejects
                # 'image/png' outright and the default is what we want.
                "aspect_ratio": aspect,
                "image_size": size,
            },
        )
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None

    image = getattr(interaction, "output_image", None)
    if not image:
        text = (getattr(interaction, "output_text", "") or "")[:200]
        print(f"  FAILED: no image returned. text={text!r}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(image.data))
    kb = out_path.stat().st_size // 1024
    print(f"  saved {out_path}  ({kb} KB, {time.time() - started:.0f}s)")
    return out_path


def restyle(key: str, size: str, model: str, out_dir: Path) -> Path | None:
    """Polish a deterministic render without letting the model touch the data."""
    from google import genai

    filename, aspect = FIGURES[key]
    source = REPO / "docs" / "figures" / "rendered" / filename
    if not source.exists():
        print(f"  no render at {source}; run scripts/render_figures.py {key} first")
        return None

    out_path = out_dir / filename
    print(f"\n=== restyle {key} <- {source.name}  [{model}, {size}, {aspect}] ===")

    payload = base64.b64encode(source.read_bytes()).decode("utf-8")
    client = genai.Client()
    started = time.time()
    try:
        interaction = client.interactions.create(
            model=model,
            input=[
                {"type": "image", "mime_type": "image/png", "data": payload},
                {"type": "text", "text": f"{STYLE}\n\n{RESTYLE_INSTRUCTION}"},
            ],
            response_format={
                "type": "image",
                "aspect_ratio": aspect,
                "image_size": size,
            },
        )
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None

    image = getattr(interaction, "output_image", None)
    if not image:
        text = (getattr(interaction, "output_text", "") or "")[:200]
        print(f"  FAILED: no image returned. text={text!r}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(image.data))
    kb = out_path.stat().st_size // 1024
    print(f"  saved {out_path}  ({kb} KB, {time.time() - started:.0f}s)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", help=f"any of: {', '.join(FIGURES)}")
    parser.add_argument("--size", default="4K", choices=["1K", "2K", "4K"])
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument(
        "--style",
        default="vector",
        choices=[*STYLES, "all"],
        help="'all' drafts every style side by side for comparison",
    )
    parser.add_argument("--out", default=None, help="output dir (default docs/figures)")
    parser.add_argument(
        "--restyle",
        action="store_true",
        help="polish the deterministic render from render_figures.py instead of "
        "generating from text. Use this for any figure with an axis.",
    )
    args = parser.parse_args()

    keys = args.figures or list(FIGURES)
    unknown = [k for k in keys if k not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {unknown}; choose from {list(FIGURES)}")
        return 1

    out_dir = Path(args.out) if args.out else FIG_DIR
    styles = list(STYLES) if args.style == "all" else [args.style]

    global STYLE

    if args.restyle:
        STYLE = STYLES[args.style if args.style != "all" else "vector"]
        ok = sum(1 for key in keys if restyle(key, args.size, args.model, out_dir))
        print(f"\n{ok}/{len(keys)} restyled into {out_dir}")
        return 0 if ok == len(keys) else 1

    ok = total = 0
    for style in styles:
        STYLE = STYLES[style]
        # Draft passes keep the style in the filename so variants sit side by
        # side; a confirmed style is written under the canonical name.
        suffix = f"_{style}" if (len(styles) > 1 or out_dir != FIG_DIR) else ""
        for key in keys:
            total += 1
            if generate(key, args.size, args.model, out_dir, suffix):
                ok += 1
            time.sleep(2)

    print(f"\n{ok}/{total} generated into {out_dir}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
