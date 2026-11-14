"""A knowledge-graph layer for out-of-distribution recall and self-repair."""
from meta_evolver.knowledge.graph import Edge, KnowledgeGraph, Node, node_id, slug
from meta_evolver.knowledge.ingest import (
    GraphIngestor,
    action_verb,
    extract_concepts,
    failure_signature,
)
from meta_evolver.knowledge.lifting import (
    LiftReport,
    induce_concept_links,
    retract_lifted,
    winning_sets,
)
from meta_evolver.knowledge.ontology import (
    Ontology,
    RelationSpec,
    RepairReport,
    Violation,
    agent_ontology,
    health,
    repair,
    validate,
)
from meta_evolver.knowledge.reasoner import (
    DEFAULT_CHAINS,
    Explanation,
    PropertyChain,
    ReasonReport,
    explain,
    reason,
    retract_derived,
)
from meta_evolver.knowledge.retrieval import GraphMemory, Recommendation

__all__ = [
    "DEFAULT_CHAINS",
    "Explanation",
    "LiftReport",
    "PropertyChain",
    "ReasonReport",
    "explain",
    "induce_concept_links",
    "reason",
    "retract_derived",
    "retract_lifted",
    "winning_sets",
    "Edge",
    "GraphIngestor",
    "GraphMemory",
    "KnowledgeGraph",
    "Node",
    "Ontology",
    "Recommendation",
    "RelationSpec",
    "RepairReport",
    "Violation",
    "action_verb",
    "agent_ontology",
    "extract_concepts",
    "failure_signature",
    "health",
    "node_id",
    "repair",
    "slug",
    "validate",
]
