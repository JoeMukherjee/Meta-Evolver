"""Offline OOD-recall report on real ALFWorld run artifacts.

    python scripts/ood_recall_report.py [--runs DIR] [--out FILE] [--seed N]

Honesty note, printed in the report too: the corpus traces record NO memory
citations per episode (only the later eval episodes do, by title, and those
carry no instruction). ``evaluate_ood_recall`` needs "which memories were in
the prompt when the task succeeded", so it cannot run as designed. This script
runs the closest honest variant and labels it PROXY:

* A training trajectory's "cited" set = memories induced from that very task.
* A held-out trajectory's gold = memories induced from *training* tasks of the
  same task family (regex-derived, see ``family``). Candidate memories are
  restricted to those induced from training tasks, so nothing from a test task
  leaks into retrieval.
No citation is invented beyond that derivation; the numbers measure
"does a retriever surface same-family knowledge", not "did the agent use it".
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LANGSMITH_TRACING", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evolver.eval.loaders import (  # noqa: E402
    discover_runs,
    load_corpus_trajectories,
    load_eval_citations,
    load_memory_bank,
)
from meta_evolver.eval.ood_recall import evaluate_ood_recall  # noqa: E402

DEFAULT_OUT = "ood_report.json"  # relative to the working directory; override with --out
KS = (1, 3, 5)
NAMES = ("similarity", "graph", "fused")
BUCKETS = ("in-distribution", "shifted", "far-OOD")


def family(instruction: str) -> str:
    s = instruction.lower()
    if re.search(r"\b(clean|heat|cool|hot|cold)\b", s):
        return "transform-then-place"
    if re.search(r"\b(two|both)\b", s):
        return "pick-two"
    if re.search(r"\b(look at|examine|under)\b", s):
        return "look-under-light"
    return "pick-and-place"


def build_proxy(trajs, meta, train_keys, test_keys):
    """Return (train, test, candidate_texts_ids) with proxy citations set."""
    by_key: dict[str, list[str]] = {}
    for mid, m in meta.items():
        by_key.setdefault(m["task_key"], []).append(mid)
    fam_of_key = {t.task_id: family(t.instruction) for t in trajs if t.instruction}
    train, test = [], []
    for t in trajs:
        if t.task_id in train_keys:
            train.append(t.model_copy(update={"retrieved_memory_ids": by_key.get(t.task_id, [])}))
        elif t.task_id in test_keys:
            gold = [
                m for k in train_keys if fam_of_key.get(k) == fam_of_key.get(t.task_id) for m in by_key.get(k, [])
            ]
            test.append(t.model_copy(update={"retrieved_memory_ids": gold}))
    return train, test


def pooled(folds: list[dict]) -> dict:
    """Pool per-fold results by reconstructing hit counts (hits = rate * n)."""
    agg = {b: {"n": 0, **{n: 0.0 for n in NAMES}} for b in (*BUCKETS, "all")}
    for f in folds:
        rows = {**f["by_shift"], "all": {"n": f["n"], **f["recall"]}}
        for b, r in rows.items():
            n = r["n"]
            agg[b]["n"] += n
            for name in NAMES:
                agg[b][name] += (r[name] or 0.0) * n
    return {
        b: {"n": a["n"], **{nm: (round(a[nm] / a["n"], 3) if a["n"] else None) for nm in NAMES}}
        for b, a in agg.items()
    }


def run_condition(trajs, texts, meta, seed):
    keys = sorted({t.task_id for t in trajs if t.usable and t.success})
    out: dict = {"n_tasks": len(keys), "protocols": {}}
    if len(keys) < 2:
        out["error"] = "fewer than 2 usable successful tasks; no task-disjoint split possible"
        return out

    def evaluate(train_keys, test_keys):
        train, test = build_proxy(trajs, meta, set(train_keys), set(test_keys))
        cand = {m: x for m, x in texts.items() if meta[m]["task_key"] in train_keys}
        res = {}
        for k in KS:
            res[k] = evaluate_ood_recall(train, test, cand, k=k)
        return res, len(train), len(test)

    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n_test = max(1, len(keys) // 2)
    test_keys, train_keys = shuffled[:n_test], shuffled[n_test:]
    assert not set(test_keys) & set(train_keys)
    res, ntr, nte = evaluate(train_keys, test_keys)
    out["protocols"]["seeded_split"] = {
        "seed": seed, "train_tasks": len(train_keys), "test_tasks": len(test_keys),
        "train_traj": ntr, "test_traj": nte, "by_k": {str(k): r for k, r in res.items()},
    }
    folds = {k: [] for k in KS}
    for held in keys:
        r, _, _ = evaluate([x for x in keys if x != held], [held])
        for k in KS:
            folds[k].append(r[k])
    out["protocols"]["leave_one_task_out"] = {"by_k": {str(k): pooled(folds[k]) for k in KS}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[3] / "runs"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = discover_runs(a.runs)
    trajs, texts, meta = [], {}, {}
    stats: Counter = Counter()
    citations = Counter()
    for run in runs:
        tr, st = load_corpus_trajectories(run / "corpus" / "traces.jsonl")
        stats.update({f"traj_{k}": v for k, v in st.items()})
        trajs += [t for t in tr if t.usable]  # infra-error episodes are not signal
        stats["traj_skipped_unusable"] += sum(1 for t in tr if not t.usable)
        for bank in sorted((run / "banks").glob("*_full.jsonl")):
            tx, mt, bs = load_memory_bank(bank, condition=bank.stem.removesuffix("_full"))
            texts.update(tx), meta.update(mt)
            stats.update({f"bank_{k}": v for k, v in bs.items()})
        for ev in sorted(run.glob("eval/round*/*.jsonl")):
            rows, _ = load_eval_citations(ev)
            citations["eval_episodes"] += len(rows)
            citations["eval_episodes_with_citations"] += sum(1 for r in rows if r["retrieved_titles"])
    # Distinct (task, rollout) across runs; the two runs reuse the same tasks.
    seen, uniq = set(), []
    for t in trajs:
        key = (t.task_id, t.rollout_index)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    trajs = uniq

    report = {
        "runs_found": len(runs),
        "load_stats": dict(stats),
        "trajectories_usable_unique": len(trajs),
        "distinct_tasks": len({t.task_id for t in trajs}),
        "success_rate": round(sum(t.success for t in trajs) / len(trajs), 3) if trajs else None,
        "memories": len(texts),
        "corpus_records_memory_citations": False,
        "eval_citations": dict(citations),
        "label": "PROXY: gold = memories induced from same-family TRAINING tasks; not real citations",
        "conditions": {},
    }
    for cond in sorted({m.split("/", 1)[0] for m in texts}):
        ct = {m: x for m, x in texts.items() if m.startswith(cond + "/")}
        cm = {m: meta[m] for m in ct}
        report["conditions"][cond] = {"n_memories": len(ct), **run_condition(trajs, ct, cm, a.seed)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"runs={len(runs)} usable_traj={len(trajs)} tasks={report['distinct_tasks']} memories={len(texts)}")
    print("corpus memory citations recorded: NO  -> PROXY labels (same-family gold)")
    for cond, c in report["conditions"].items():
        for proto, p in c.get("protocols", {}).items():
            for k, res in p["by_k"].items():
                rows = res["by_shift"] if "by_shift" in res else {b: res[b] for b in (*BUCKETS, "all")}
                if "by_shift" in res:
                    rows = {**res["by_shift"], "all": {"n": res["n"], **res["recall"]}}
                for b, r in rows.items():
                    print(f"{cond:5s} {proto:20s} k={k} {b:16s} n={r['n']:3d} "
                          + " ".join(f"{n}={r[n]}" for n in NAMES))
        if "error" in c:
            print(cond, c["error"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
