"""OpenTelemetry and MLflow compatible step-level tracing for agent rollouts.

Provides structured trace collection for agent execution nodes (prepare, think,
act, adapt, finalize, induce, optimize_prompt). Spans record latencies,
intermediate token counts, prompt inputs, tool calls, and error states.

Traces can be exported to:
- **MLflow Tracing format** (compatible with MLflow 2.16+ / 3.x Trace UI)
- **OpenTelemetry JSON** (compatible with Arize Phoenix, OTel collectors, Jaeger)
- **Local JSONL** (for offline auditing and causal graph linking)
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceSpan(BaseModel):
    """A single timed span within an episode or generation trace."""

    model_config = ConfigDict(extra="allow")
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: str | None = None
    name: str = ""
    node_name: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "OK"  # "OK" | "ERROR" | "UNSET"
    error: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    def finish(self, status: str = "OK", error: str = "") -> None:
        """Mark span complete and compute duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = "ERROR" if error else status
        self.error = error

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )


class TelemetryTracer:
    """Collects and exports hierarchical trace trees across agent runs."""

    def __init__(self, trace_id: str | None = None, task_id: str = "", run_id: str = ""):
        self.trace_id: str = trace_id or f"tr_{uuid.uuid4().hex[:12]}"
        self.task_id: str = task_id
        self.run_id: str = run_id
        self.spans: list[TraceSpan] = []
        self._span_map: dict[str, TraceSpan] = {}
        self._active_span_stack: list[str] = []

    def start_span(
        self,
        name: str,
        node_name: str = "",
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Create and start a new child span."""
        parent_id = parent_span_id
        if parent_id is None and self._active_span_stack:
            parent_id = self._active_span_stack[-1]

        span = TraceSpan(
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            name=name,
            node_name=node_name or name,
            attributes=attributes or {},
        )
        self.spans.append(span)
        self._span_map[span.span_id] = span
        self._active_span_stack.append(span.span_id)
        return span

    def end_span(
        self,
        span_id: str | None = None,
        status: str = "OK",
        error: str = "",
    ) -> TraceSpan | None:
        """Finish an active span."""
        target_id = span_id or (self._active_span_stack[-1] if self._active_span_stack else None)
        if not target_id or target_id not in self._span_map:
            return None

        span = self._span_map[target_id]
        span.finish(status=status, error=error)

        if self._active_span_stack and self._active_span_stack[-1] == target_id:
            self._active_span_stack.pop()

        return span

    @contextmanager
    def span(
        self,
        name: str,
        node_name: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        """Synchronous context manager for span timing."""
        span_obj = self.start_span(name=name, node_name=node_name, attributes=attributes)
        error_str = ""
        try:
            yield span_obj
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.end_span(span_obj.span_id, error=error_str)

    @asynccontextmanager
    async def aspan(
        self,
        name: str,
        node_name: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceSpan]:
        """Asynchronous context manager for span timing."""
        span_obj = self.start_span(name=name, node_name=node_name, attributes=attributes)
        error_str = ""
        try:
            yield span_obj
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.end_span(span_obj.span_id, error=error_str)

    # -- Exporters ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to structured dictionary."""
        total_duration = sum(s.duration_ms for s in self.spans if s.parent_span_id is None)
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "total_spans": len(self.spans),
            "total_duration_ms": total_duration,
            "spans": [s.model_dump() for s in self.spans],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_mlflow_trace(self) -> dict[str, Any]:
        """Format trace structure matching MLflow Tracing schema."""
        root_spans = [s for s in self.spans if s.parent_span_id is None]
        root_name = root_spans[0].name if root_spans else "Episode"
        return {
            "request_id": self.trace_id,
            "experiment_name": "Meta-Evolver",
            "root_name": root_name,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "timestamp_ms": int(self.spans[0].start_time * 1000) if self.spans else 0,
            "execution_time_ms": sum(s.duration_ms for s in root_spans),
            "status": "ERROR" if any(s.status == "ERROR" for s in self.spans) else "OK",
            "spans": [
                {
                    "name": s.name,
                    "span_id": s.span_id,
                    "parent_id": s.parent_span_id,
                    "start_time_unix_nano": int(s.start_time * 1e9),
                    "end_time_unix_nano": int((s.end_time or s.start_time) * 1e9),
                    "status": s.status,
                    "attributes": s.attributes,
                    "events": s.events,
                }
                for s in self.spans
            ],
        }

    def to_opentelemetry_format(self) -> dict[str, Any]:
        """Format trace structure matching OpenTelemetry JSON representation."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "meta-evolver"}},
                            {"key": "task.id", "value": {"stringValue": self.task_id}},
                            {"key": "run.id", "value": {"stringValue": self.run_id}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": self.trace_id,
                                    "spanId": s.span_id,
                                    "parentSpanId": s.parent_span_id or "",
                                    "name": s.name,
                                    "kind": 1,  # INTERNAL
                                    "startTimeUnixNano": int(s.start_time * 1e9),
                                    "endTimeUnixNano": int((s.end_time or s.start_time) * 1e9),
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in s.attributes.items()
                                    ],
                                    "status": {"code": 2 if s.status == "ERROR" else 1},
                                }
                                for s in self.spans
                            ]
                        }
                    ],
                }
            ]
        }

    def save_to_file(self, path: Path | str) -> None:
        """Export trace JSON to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())
