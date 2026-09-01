"""The causal graph of an evolution run.

What this records, and why a graph rather than a table.

The evolution loop rewrites three pieces of shared scaffold -- the prompt, the
memory bank, the curriculum -- and the interesting questions about a run are
all about *provenance*:

* Which episodes produced the memory that later carried a task?
* Did the prompt that got adopted in generation 3 actually help in 4, or did a
  memory pruned at the same time explain the change?
* When a task regressed, what changed between the generation that passed it
  and the one that did not?

Those are path queries. In JSONL they are joins nobody writes; as a graph they
are one Cypher line, and in the Neo4j browser they are a picture you can watch
grow while the run happens.

The schema is deliberately small -- six node labels, and edges that all mean
"this caused / produced / was used by that":

.. code-block:: text

    (:Run)-[:HAS_GENERATION]->(:Generation)
    (:Generation)-[:RAN]->(:Episode)-[:ON_TASK]->(:Task)
    (:Generation)-[:USED_PROMPT]->(:Prompt)
    (:Generation)-[:AT_LEVEL]->(:CurriculumLevel)
    (:Episode)-[:RETRIEVED]->(:Memory)         -- memory was in the prompt
    (:Episode)-[:INDUCED]->(:Memory)           -- episode produced the memory
    (:Generation)-[:PRUNED]->(:Memory)         -- and where it was dropped
    (:Prompt)-[:PROPOSED_FROM]->(:Prompt)      -- candidate lineage
    (:Generation)-[:ADOPTED]->(:Prompt)

``RETRIEVED`` and ``INDUCED`` are the two that make the memory bank's causal
history legible: one says a memory was *used*, the other says an episode
*created* it. Following both directions from a memory node answers "where did
this come from, and did it ever help" without any bookkeeping elsewhere.
"""
from __future__ import annotations

#: Constraints and indexes. Idempotent, applied on connect.
#:
#: Uniqueness is on a composite key including the run, so several runs can
#: share one database and still be queried apart -- the same reason the memory
#: bank namespaces its rows.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT me_run IF NOT EXISTS "
    "FOR (n:Run) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT me_generation IF NOT EXISTS "
    "FOR (n:Generation) REQUIRE (n.run_id, n.index) IS UNIQUE",
    "CREATE CONSTRAINT me_episode IF NOT EXISTS "
    "FOR (n:Episode) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT me_task IF NOT EXISTS "
    "FOR (n:Task) REQUIRE (n.benchmark, n.task_id) IS UNIQUE",
    "CREATE CONSTRAINT me_memory IF NOT EXISTS "
    "FOR (n:Memory) REQUIRE (n.run_id, n.id) IS UNIQUE",
    "CREATE CONSTRAINT me_prompt IF NOT EXISTS "
    "FOR (n:Prompt) REQUIRE (n.run_id, n.version) IS UNIQUE",
    "CREATE CONSTRAINT me_level IF NOT EXISTS "
    "FOR (n:CurriculumLevel) REQUIRE (n.run_id, n.level) IS UNIQUE",
    "CREATE INDEX me_episode_gen IF NOT EXISTS "
    "FOR (n:Episode) ON (n.run_id, n.generation)",
)

#: Saved queries, surfaced by ``meta-evolver graph queries`` and pasteable
#: straight into the Neo4j browser. Each answers a question the flat telemetry
#: makes you write a join for.
SAVED_QUERIES: dict[str, tuple[str, str]] = {
    "overview": (
        "The whole run: generations, the prompt each used, and its episodes.",
        """
MATCH (r:Run {id: $run_id})-[:HAS_GENERATION]->(g:Generation)
OPTIONAL MATCH (g)-[:USED_PROMPT]->(p:Prompt)
OPTIONAL MATCH (g)-[:RAN]->(e:Episode)
RETURN r, g, p, e
""".strip(),
    ),
    "memory_provenance": (
        "Where each memory came from, and every episode that later used it.",
        """
MATCH (m:Memory {run_id: $run_id})
OPTIONAL MATCH (born:Episode)-[:INDUCED]->(m)
OPTIONAL MATCH (used:Episode)-[:RETRIEVED]->(m)
RETURN m, born, used
""".strip(),
    ),
    "earned_its_slot": (
        "Memories ranked by how often episodes citing them actually passed.",
        """
MATCH (e:Episode {run_id: $run_id})-[:RETRIEVED]->(m:Memory)
WITH m, count(e) AS uses, sum(CASE WHEN e.success THEN 1 ELSE 0 END) AS wins
RETURN m.title AS memory, m.scenario AS scenario, uses, wins,
       toFloat(wins + 1) / (uses + 2) AS utility
ORDER BY utility DESC
""".strip(),
    ),
    "regressions": (
        "Tasks that passed in one generation and failed in the next, with "
        "what changed between them.",
        """
MATCH (a:Episode {run_id: $run_id, success: true})-[:ON_TASK]->(t:Task)
MATCH (b:Episode {run_id: $run_id, success: false})-[:ON_TASK]->(t)
WHERE b.generation = a.generation + 1
MATCH (ga:Generation {run_id: $run_id, index: a.generation})
MATCH (gb:Generation {run_id: $run_id, index: b.generation})
RETURN t.task_id AS task, a.generation AS passed_at, b.generation AS failed_at,
       ga.prompt_version AS prompt_before, gb.prompt_version AS prompt_after,
       ga.curriculum_level AS level_before, gb.curriculum_level AS level_after
""".strip(),
    ),
    "prompt_lineage": (
        "How the system prompt evolved, and which candidates were rejected.",
        """
MATCH (p:Prompt {run_id: $run_id})
OPTIONAL MATCH (p)-[:PROPOSED_FROM]->(parent:Prompt)
OPTIONAL MATCH (g:Generation)-[:ADOPTED]->(p)
RETURN p.version AS version, parent.version AS parent,
       p.adopted AS adopted, p.validation_score AS validation,
       g.index AS adopted_at
ORDER BY version
""".strip(),
    ),
    "scaffold_at": (
        "The complete scaffold as of one generation -- prompt, memories in "
        "play, and difficulty. Set $generation.",
        """
MATCH (g:Generation {run_id: $run_id, index: $generation})
OPTIONAL MATCH (g)-[:USED_PROMPT]->(p:Prompt)
OPTIONAL MATCH (g)-[:AT_LEVEL]->(l:CurriculumLevel)
OPTIONAL MATCH (g)-[:RAN]->(:Episode)-[:RETRIEVED]->(m:Memory)
RETURN g, p, l, collect(DISTINCT m) AS memories_in_play
""".strip(),
    ),
}
