"""Core data types and schemas for Meta-Evolver."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class StepRecord(BaseModel):
    step_idx: int
    action: str
    observation: str
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

class Trajectory(BaseModel):
    task_id: str
    instruction: str
    steps: List[StepRecord] = Field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)

class MemoryItem(BaseModel):
    id: str
    title: str
    task_pattern: str
    strategy_rule: str
    source_success_rate: float = 1.0
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)
