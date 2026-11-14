"""An embedded, time-aware labeled property graph (LPG).

Why this exists next to the memory bank
---------------------------------------
:class:`~meta_evolver.memory.bank.ReasoningMemoryBank` retrieves by text
similarity. That is the right tool in distribution and exactly the wrong one
out of it: a task phrased with different words from every task the agent has
seen has no near neighbour, so the bank returns the nearest *wrong* strategy
(the "retrieval trap" the README describes). A graph retrieves by *structure*
instead -- the tools a task needs, the concepts it involves, the failure modes
it has hit -- and structure survives a change of surface wording.

Design, deliberately small
--------------------------
* Pure Python + SQLite. No server, no compiled wheel (this project already
  runs on machines where a policy blocks native extensions), and a graph of
  tens of thousands of edges answers a 2-hop expansion in milliseconds.
* **Time-aware edges.** An edge has ``valid_from``/``valid_to`` in
  *generations*. A fact that stops being true is *invalidated*, not deleted,
  so "what did the agent believe at generation 4" stays answerable and a
  wrongly-retired edge can be audited. This is Graphiti's bi-temporal idea
  reduced to the one clock this engine has.
* **Edges carry evidence.** ``wins``/``uses`` are a Beta(1,1) posterior --
  the same accrual the memory bank uses -- so an ``applies_to`` edge from a
  strategy to a domain *is* its transfer evidence for that domain.
* RDF interop is one-way and cheap: :meth:`to_ntriples`.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG.sub("-", str(text).lower()).strip("-")[:80] or "x"


def node_id(label: str, name: str) -> str:
    """Stable identity: label plus normalised name, so re-ingesting the same
    entity from another episode lands on the same node."""
    return f"{label}:{slug(name)}"


@dataclass(frozen=True)
class Edge:
    id: int
    src: str
    rel: str
    dst: str
    props: dict[str, Any]
    valid_from: int
    valid_to: int | None
    wins: float
    uses: float

    @property
    def active(self) -> bool:
        return self.valid_to is None

    @property
    def confidence(self) -> float:
        """Beta(1,1) posterior mean -- an unobserved edge sits at 0.5."""
        return (self.wins + 1.0) / (self.uses + 2.0)


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    name: str
    props: dict[str, Any]
    created_gen: int
    updated_gen: int


@dataclass
class Reach:
    """One node reached by :meth:`KnowledgeGraph.traverse`."""

    node: str
    distance: int
    score: float
    path: list[str] = field(default_factory=list)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
    id TEXT PRIMARY KEY, label TEXT NOT NULL, name TEXT NOT NULL,
    props TEXT NOT NULL DEFAULT '{}', created_gen INTEGER NOT NULL DEFAULT 0,
    updated_gen INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS edges(
    id INTEGER PRIMARY KEY AUTOINCREMENT, src TEXT NOT NULL, rel TEXT NOT NULL,
    dst TEXT NOT NULL, props TEXT NOT NULL DEFAULT '{}',
    valid_from INTEGER NOT NULL DEFAULT 0, valid_to INTEGER,
    wins REAL NOT NULL DEFAULT 0, uses REAL NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_nodes_label ON nodes(label);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src, rel);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst, rel);
"""


class KnowledgeGraph:
    """SQLite-backed labeled property graph. ``path=None`` keeps it in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across the event loop's worker threads; the
        # lock keeps writes serial (SQLite would anyway, but a torn
        # read-modify-write in add_edge would not be caught by it).
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    # -- nodes ---------------------------------------------------------------

    def upsert_node(
        self, label: str, name: str, props: dict[str, Any] | None = None, generation: int = 0
    ) -> str:
        nid = node_id(label, name)
        with self._lock:
            row = self._db.execute("SELECT props FROM nodes WHERE id=?", (nid,)).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO nodes(id,label,name,props,created_gen,updated_gen) VALUES(?,?,?,?,?,?)",
                    (nid, label, name, json.dumps(props or {}), generation, generation),
                )
            else:
                merged = {**json.loads(row["props"]), **(props or {})}
                self._db.execute(
                    "UPDATE nodes SET props=?, updated_gen=? WHERE id=?",
                    (json.dumps(merged), generation, nid),
                )
            self._db.commit()
        return nid

    def get_node(self, nid: str) -> Node | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM nodes WHERE id=?", (nid,)).fetchone()
        return _node(row) if row else None

    def nodes(self, label: str | None = None) -> list[Node]:
        with self._lock:
            rows = (
                self._db.execute("SELECT * FROM nodes WHERE label=?", (label,)).fetchall()
                if label
                else self._db.execute("SELECT * FROM nodes").fetchall()
            )
        return [_node(r) for r in rows]

    def relabel(self, nid: str, new_label: str) -> str:
        """Change a node's label (and therefore its id), re-pointing its edges."""
        node = self.get_node(nid)
        if node is None or node.label == new_label:
            return nid
        new_id = node_id(new_label, node.name)
        with self._lock:
            if self.get_node(new_id) is None:
                self._db.execute(
                    "UPDATE nodes SET id=?, label=? WHERE id=?", (new_id, new_label, nid)
                )
                self._db.execute("UPDATE edges SET src=? WHERE src=?", (new_id, nid))
                self._db.execute("UPDATE edges SET dst=? WHERE dst=?", (new_id, nid))
                self._db.commit()
            else:  # target already exists: merge into it
                self.merge_nodes(new_id, nid)
        return new_id

    def merge_nodes(self, keep: str, drop: str) -> None:
        """Fold ``drop`` into ``keep``: re-point edges, union props, collapse
        duplicate edges (pooling their evidence), delete ``drop``."""
        if keep == drop:
            return
        with self._lock:
            a, b = self.get_node(keep), self.get_node(drop)
            if a is None or b is None:
                return
            self._db.execute(
                "UPDATE nodes SET props=?, created_gen=MIN(created_gen,?) WHERE id=?",
                (json.dumps({**b.props, **a.props}), b.created_gen, keep),
            )
            self._db.execute("UPDATE edges SET src=? WHERE src=?", (keep, drop))
            self._db.execute("UPDATE edges SET dst=? WHERE dst=?", (keep, drop))
            self._db.execute("DELETE FROM nodes WHERE id=?", (drop,))
            self._collapse_edges()
            self._db.commit()

    def _collapse_edges(self) -> None:
        # Self-loops created by a merge carry no information.
        self._db.execute("DELETE FROM edges WHERE src = dst")
        groups = self._db.execute(
            "SELECT src,rel,dst,MIN(id) AS keep, SUM(wins) AS w, SUM(uses) AS u, COUNT(*) AS n "
            "FROM edges WHERE valid_to IS NULL GROUP BY src,rel,dst HAVING n > 1"
        ).fetchall()
        for g in groups:
            self._db.execute("UPDATE edges SET wins=?, uses=? WHERE id=?", (g["w"], g["u"], g["keep"]))
            self._db.execute(
                "DELETE FROM edges WHERE src=? AND rel=? AND dst=? AND valid_to IS NULL AND id<>?",
                (g["src"], g["rel"], g["dst"], g["keep"]),
            )

    # -- edges ---------------------------------------------------------------

    def add_edge(
        self,
        src: str,
        rel: str,
        dst: str,
        generation: int = 0,
        props: dict[str, Any] | None = None,
        functional: bool = False,
        evidence: bool | None = None,
    ) -> int:
        """Assert ``src -rel-> dst``.

        Re-asserting an active edge strengthens it instead of duplicating it:
        ``evidence=True/False`` records a win/loss on it (``None`` records
        nothing, so structural edges are not given a false 0.5-with-uses).
        ``functional=True`` first invalidates any other active ``src -rel->``
        edge -- "a task is in exactly one domain" -- at ``generation``.
        """
        with self._lock:
            if functional:
                self._db.execute(
                    "UPDATE edges SET valid_to=? WHERE src=? AND rel=? AND dst<>? AND valid_to IS NULL",
                    (generation, src, rel, dst),
                )
            row = self._db.execute(
                "SELECT id FROM edges WHERE src=? AND rel=? AND dst=? AND valid_to IS NULL",
                (src, rel, dst),
            ).fetchone()
            if row is None:
                cur = self._db.execute(
                    "INSERT INTO edges(src,rel,dst,props,valid_from) VALUES(?,?,?,?,?)",
                    (src, rel, dst, json.dumps(props or {}), generation),
                )
                eid = int(cur.lastrowid)
            else:
                eid = int(row["id"])
            if evidence is not None:
                self._db.execute(
                    "UPDATE edges SET uses=uses+1, wins=wins+? WHERE id=?",
                    (1.0 if evidence else 0.0, eid),
                )
            self._db.commit()
        return eid

    def invalidate(self, edge_id: int, generation: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE edges SET valid_to=? WHERE id=? AND valid_to IS NULL", (generation, edge_id)
            )
            self._db.commit()

    def delete_edge(self, edge_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM edges WHERE id=?", (edge_id,))
            self._db.commit()

    def set_edge(self, edge_id: int, src: str | None = None, dst: str | None = None, rel: str | None = None) -> None:
        with self._lock:
            e = self._edge(edge_id)
            if e is None:
                return
            self._db.execute(
                "UPDATE edges SET src=?, dst=?, rel=? WHERE id=?",
                (src or e.src, dst or e.dst, rel or e.rel, edge_id),
            )
            self._db.commit()

    def _edge(self, edge_id: int) -> Edge | None:
        row = self._db.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
        return _edge(row) if row else None

    def edges(
        self,
        src: str | None = None,
        rel: str | None = None,
        dst: str | None = None,
        at: int | None = None,
        include_inactive: bool = False,
    ) -> list[Edge]:
        """Edges matching the given ends. ``at`` answers "as of generation N";
        otherwise only currently-active edges are returned."""
        clauses, args = [], []
        for col, val in (("src", src), ("rel", rel), ("dst", dst)):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        if at is not None:
            clauses.append("valid_from<=? AND (valid_to IS NULL OR valid_to>?)")
            args += [at, at]
        elif not include_inactive:
            clauses.append("valid_to IS NULL")
        sql = "SELECT * FROM edges" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        with self._lock:
            return [_edge(r) for r in self._db.execute(sql, args).fetchall()]

    # -- traversal -----------------------------------------------------------

    def neighbors(self, nid: str, rels: Iterable[str] | None = None, direction: str = "both") -> list[tuple[Edge, str]]:
        """``(edge, other_end)`` pairs for active edges touching ``nid``."""
        want = set(rels) if rels is not None else None
        out: list[tuple[Edge, str]] = []
        if direction in ("out", "both"):
            out += [(e, e.dst) for e in self.edges(src=nid) if want is None or e.rel in want]
        if direction in ("in", "both"):
            out += [(e, e.src) for e in self.edges(dst=nid) if want is None or e.rel in want]
        return out

    def traverse(
        self,
        seeds: Sequence[str],
        hops: int = 2,
        rels: Iterable[str] | None = None,
        direction: str = "both",
        decay: float = 0.5,
        max_nodes: int = 500,
    ) -> dict[str, Reach]:
        """Breadth-first expansion from ``seeds``.

        A node's score is the best path product of edge confidence and
        ``decay**distance`` -- so a 1-hop neighbour over a well-evidenced edge
        outranks a 2-hop one, and an edge the agent has repeatedly seen fail
        drags everything behind it down. ``max_nodes`` bounds the cost of a
        hub (a ``Domain`` node touched by every task).
        """
        reach: dict[str, Reach] = {s: Reach(s, 0, 1.0, [s]) for s in seeds if self.get_node(s)}
        queue = deque(reach)
        while queue and len(reach) < max_nodes:
            cur = queue.popleft()
            r = reach[cur]
            if r.distance >= hops:
                continue
            for edge, other in self.neighbors(cur, rels, direction):
                score = r.score * edge.confidence * decay
                prev = reach.get(other)
                if prev is None:
                    reach[other] = Reach(other, r.distance + 1, score, [*r.path, other])
                    queue.append(other)
                elif score > prev.score:
                    prev.score, prev.path = score, [*r.path, other]
        return reach

    # -- housekeeping --------------------------------------------------------

    def degree(self, nid: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE (src=? OR dst=?) AND valid_to IS NULL", (nid, nid)
            ).fetchone()
        return int(row["n"])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            active = self._db.execute("SELECT COUNT(*) FROM edges WHERE valid_to IS NULL").fetchone()[0]
            total = self._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            labels = dict(self._db.execute("SELECT label, COUNT(*) FROM nodes GROUP BY label").fetchall())
        return {"nodes": n, "edges_active": active, "edges_total": total, "by_label": labels}

    def to_ntriples(self, base: str = "urn:meta-evolver:") -> str:
        """Active edges as N-Triples, for any RDF store (e.g. to run OWL RL
        inference over them offline). Node labels become ``rdf:type``."""
        lines = []
        for n in self.nodes():
            lines.append(f"<{base}{n.id}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{base}class/{n.label}> .")
        for e in self.edges():
            lines.append(f"<{base}{e.src}> <{base}rel/{e.rel}> <{base}{e.dst}> .")
        return "\n".join(lines) + ("\n" if lines else "")

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _node(row: sqlite3.Row) -> Node:
    return Node(row["id"], row["label"], row["name"], json.loads(row["props"]), row["created_gen"], row["updated_gen"])


def _edge(row: sqlite3.Row) -> Edge:
    return Edge(
        row["id"], row["src"], row["rel"], row["dst"], json.loads(row["props"]),
        row["valid_from"], row["valid_to"], row["wins"], row["uses"],
    )
