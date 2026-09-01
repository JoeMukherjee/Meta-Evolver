"""Run telemetry: streaming JSONL, OpenTelemetry/MLflow tracing, and summaries."""
from meta_evolver.telemetry.engine import TelemetryEngine
from meta_evolver.telemetry.tracer import TelemetryTracer, TraceSpan

__all__ = ["TelemetryEngine", "TelemetryTracer", "TraceSpan"]
