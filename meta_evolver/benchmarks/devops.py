"""DevOps incident benchmark -- a self-contained SRE simulator.

Five production incidents with deterministic ground truth. Each requires the
same four-beat discipline -- diagnose, patch, restart, verify -- and each
punishes a different shortcut:

* ``db_pool``     -- the symptom is in one service, the cause in another.
* ``jwt_auth``    -- two services disagree; the logs name which is stale.
* ``cache_oom``   -- two config keys are wrong; fixing one is not enough.
* ``rate_limit``  -- the obvious fix (raise the limit) is wrong; the caller is.
* ``disk_pressure`` -- the loudest log line is a symptom, not the cause.

The last two exist because an agent can pass the first three by pattern
matching "find the number, make it bigger". They are the tasks where a
memorized procedure actively misleads, which is exactly what the adaptive
controller and the anti-pattern memories are for.

Tools are declared on the environment, not in a global registry. An earlier
version handed the agent one shared list of eighteen tools -- web search, code
execution, document generation -- for a task that needs seven, and episodes
were routinely lost to probing irrelevant capabilities.
"""
from __future__ import annotations

import copy
from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.registry import register_benchmark
from meta_evolver.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

TASKS: dict[str, dict[str, Any]] = {
    "db_pool": {
        "title": "Payment gateway returning 500s",
        "instruction": (
            "Checkout requests to the payment service are failing with 500 Internal "
            "Server Error. Find the root cause, fix it, and prove the service is healthy."
        ),
        "difficulty": 0.3,
        "services": {
            "payment-gateway": {
                "status": "degraded",
                "config": {"db_proxy_url": "db-proxy.internal:5432", "timeout_ms": 3000},
                "logs": [
                    "[INFO] POST /payments/charge",
                    "[ERROR] pool timeout: could not acquire connection within 3000ms",
                    "[ERROR] 500 returned to client",
                ],
            },
            "db-proxy": {
                "status": "active",
                "config": {"max_connections": 5, "idle_timeout": 30},
                "logs": ["[WARN] connections 5/5, queue depth 42 and growing"],
            },
        },
        "tables": {
            "connection_pool_metrics": [
                {"pool": "payment_pool", "max": 5, "in_use": 5, "queue_depth": 42}
            ]
        },
        "goal": {
            "service": "db-proxy",
            "key": "max_connections",
            "min_value": 20,
            "endpoint": "payments/charge",
        },
    },
    "jwt_auth": {
        "title": "Every authenticated request returns 401",
        "instruction": (
            "All authenticated client requests are rejected with 401 Unauthorized. "
            "Identify the mismatch, correct it, and verify."
        ),
        "difficulty": 0.35,
        "services": {
            "auth-service": {
                "status": "degraded",
                "config": {"active_key_version": "v1", "token_ttl_hours": 24},
                "logs": [
                    "[WARN] verification failed: header key_id='v2', local active key 'v1'",
                    "[WARN] 401 Unauthorized",
                ],
            },
            "gateway": {
                "status": "active",
                "config": {"expected_key_version": "v2"},
                "logs": ["[INFO] routing auth requests to auth-service.internal"],
            },
        },
        "tables": {
            "auth_key_store": [
                {"key_id": "v1", "status": "deprecated", "revoked_at": "2026-08-01"},
                {"key_id": "v2", "status": "active", "created_at": "2026-08-01"},
            ]
        },
        "goal": {
            "service": "auth-service",
            "key": "active_key_version",
            "expected": "v2",
            "endpoint": "auth/validate",
        },
    },
    "cache_oom": {
        "title": "Cache worker out of memory, latency at 5s",
        "instruction": (
            "cache-worker is rejecting writes and p99 latency has reached 5000ms. "
            "Diagnose it, apply a fix that will hold, and verify."
        ),
        "difficulty": 0.6,
        "services": {
            "cache-worker": {
                "status": "critical",
                "config": {
                    "max_memory_mb": 128,
                    "ttl_seconds": 0,
                    "eviction_policy": "noeviction",
                },
                "logs": [
                    "[ERROR] OOM command not allowed: memory limit 128MB reached",
                    "[WARN] eviction_policy=noeviction and ttl=0: keys never expire",
                ],
            }
        },
        "tables": {
            "cache_metrics": [
                {"cluster": "main", "used_mb": 128, "max_mb": 128, "evictions_1h": 0}
            ]
        },
        # Two keys must change. Raising memory alone leaves a cache that still
        # never expires anything and will refill within the hour.
        "goal": {
            "service": "cache-worker",
            "key": "ttl_seconds",
            "min_value": 300,
            "also": {"eviction_policy": ("allkeys-lru", "volatile-lru", "allkeys-lfu")},
            "endpoint": "cache/health",
        },
    },
    "rate_limit": {
        "title": "Public API shedding load at 429",
        "instruction": (
            "The public API is returning 429 Too Many Requests to legitimate users. "
            "Find why, fix it correctly, and verify."
        ),
        "difficulty": 0.8,
        "services": {
            "api-gateway": {
                "status": "degraded",
                "config": {"rate_limit_rpm": 600, "burst": 50},
                "logs": [
                    "[WARN] 429 for tenant=acme (14,900 rpm observed, limit 600)",
                    "[INFO] all other tenants below 40 rpm",
                ],
            },
            "sync-worker": {
                "status": "active",
                "config": {"poll_interval_ms": 20, "batch_size": 1},
                "logs": [
                    "[INFO] polling /v1/items every 20ms for tenant=acme",
                    "[INFO] retry_on_empty=true",
                ],
            },
        },
        "tables": {
            "request_log": [
                {"tenant": "acme", "source": "sync-worker", "rpm": 14900},
                {"tenant": "globex", "source": "web", "rpm": 38},
            ]
        },
        # Raising the gateway limit "fixes" the 429 and leaves a worker
        # hammering the API 250x faster than it needs to. The fix is upstream.
        "goal": {
            "service": "sync-worker",
            "key": "poll_interval_ms",
            "min_value": 1000,
            "endpoint": "v1/items",
            "wrong_fix": {
                "service": "api-gateway",
                "key": "rate_limit_rpm",
                "note": (
                    "Raising the gateway limit hides the symptom: one client is "
                    "polling 50x per second. The 429s are correct behaviour."
                ),
            },
        },
    },
    "disk_pressure": {
        "title": "Write failures across the ingest tier",
        "instruction": (
            "ingest-api is failing writes intermittently. The loudest error is not "
            "necessarily the cause. Diagnose, fix, and verify."
        ),
        "difficulty": 0.85,
        "services": {
            "ingest-api": {
                "status": "degraded",
                "config": {"write_retries": 3, "buffer_mb": 64},
                "logs": [
                    "[ERROR] write failed: No space left on device (/var/log)",
                    "[ERROR] write failed: No space left on device (/var/log)",
                    "[INFO] data volume /data at 31% used",
                ],
            },
            "log-shipper": {
                "status": "active",
                "config": {"retention_days": 3650, "compress": False},
                "logs": [
                    "[INFO] shipping to remote sink OK",
                    "[WARN] local spool 412GB across 3650 days of retention",
                ],
            },
        },
        "tables": {
            "disk_usage": [
                {"mount": "/var/log", "used_pct": 100, "owner": "log-shipper"},
                {"mount": "/data", "used_pct": 31, "owner": "ingest-api"},
            ]
        },
        # The failing service is ingest-api; the cause is log-shipper's
        # ten-year local retention filling /var/log.
        "goal": {
            "service": "log-shipper",
            "key": "retention_days",
            "max_value": 30,
            "endpoint": "ingest/health",
            "wrong_fix": {
                "service": "ingest-api",
                "key": "buffer_mb",
                "note": (
                    "Tuning the writer does not create disk space. The full mount "
                    "is /var/log, owned by log-shipper."
                ),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DevOpsIncidentEnv(ActionableEnv):
    """A microservice fleet with one injected fault and verifiable ground truth."""

    env_type = "devops_incident"

    tool_schemas = [
        tool_schema(
            "inspect_service_logs",
            "Read recent log lines from a service.",
            {
                "service_name": {"type": "string", "description": "Service to read."},
                "tail_lines": {"type": "integer", "description": "How many lines (default 20)."},
            },
            ["service_name"],
        ),
        tool_schema(
            "inspect_service_config",
            "Read the live configuration of a service.",
            {"service_name": {"type": "string"}},
            ["service_name"],
        ),
        tool_schema(
            "query_metrics",
            "Query an operational metrics table. Name the table in the query.",
            {"query": {"type": "string", "description": "e.g. 'select * from disk_usage'"}},
            ["query"],
        ),
        tool_schema(
            "patch_service_config",
            "Apply a partial config update to a service. Takes effect on restart.",
            {
                "service_name": {"type": "string"},
                "config_patch": {
                    "type": "object",
                    "description": "Keys to overwrite, e.g. {'max_connections': 50}.",
                },
            },
            ["service_name", "config_patch"],
        ),
        tool_schema(
            "restart_service",
            "Restart a service so its pending configuration takes effect.",
            {"service_name": {"type": "string"}},
            ["service_name"],
        ),
        tool_schema(
            "run_healthcheck",
            "Probe an endpoint. Returns the real status code.",
            {"endpoint": {"type": "string", "description": "e.g. 'payments/charge'"}},
            ["endpoint"],
        ),
        tool_schema(
            "submit_resolution",
            "Close the incident. Only accepted once the healthcheck passes.",
            {
                "root_cause": {"type": "string"},
                "action_taken": {"type": "string"},
            },
            ["root_cause", "action_taken"],
        ),
    ]

    def __init__(self, task_id: str = "db_pool", max_steps: int = 20) -> None:
        self.task_id = task_id if task_id in TASKS else "db_pool"
        self.max_steps = int(max_steps)
        self.spec: dict[str, Any] = {}
        self.services: dict[str, Any] = {}
        self.tables: dict[str, Any] = {}
        self.step_count = 0
        self.restarted: set[str] = set()
        self.healthcheck_passed = False
        self.submitted: dict[str, str] | None = None
        self.last_result: Any = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.spec = copy.deepcopy(TASKS[self.task_id])
        self.services = self.spec["services"]
        self.tables = self.spec["tables"]
        self.step_count = 0
        self.restarted = set()
        self.healthcheck_passed = False
        self.submitted = None
        self.last_result = None

    # -- step loop ---------------------------------------------------------

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        opts = options or {}
        requested = opts.get("task_id")
        if requested in TASKS:
            self.task_id = requested
        self._reset_state()
        return EnvResetResponse(
            observation=self.observe(),
            info={
                "task_id": self.task_id,
                "title": self.spec["title"],
                "instruction": self.spec["instruction"],
            },
        )

    def step(self, action: Action) -> EnvResponse:
        self.step_count += 1
        result = self._dispatch(action.name, action.kwargs or {})
        self.last_result = result

        evaluation = self.evaluate()
        done = self.submitted is not None
        return EnvResponse(
            observation=self.observe(),
            reward=1.0 if evaluation.success else evaluation.score,
            terminated=done,
            truncated=self.step_count >= self.max_steps,
            info={"result": result, "step": self.step_count},
        )

    def _dispatch(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_service_logs":
            service = kwargs.get("service_name", "")
            if service not in self.services:
                return self._unknown_service(service)
            tail = int(kwargs.get("tail_lines", 20) or 20)
            entry = self.services[service]
            return {"service": service, "status": entry["status"], "logs": entry["logs"][-tail:]}

        if name == "inspect_service_config":
            service = kwargs.get("service_name", "")
            if service not in self.services:
                return self._unknown_service(service)
            return {"service": service, "config": self.services[service]["config"]}

        if name == "query_metrics":
            query = str(kwargs.get("query", "")).lower()
            matched = [t for t in self.tables if t in query]
            if not matched:
                return {
                    "error": "no known table named in query",
                    "available_tables": sorted(self.tables),
                }
            return {"table": matched[0], "rows": self.tables[matched[0]]}

        if name == "patch_service_config":
            service = kwargs.get("service_name", "")
            if service not in self.services:
                return self._unknown_service(service)
            patch = kwargs.get("config_patch") or {}
            if not isinstance(patch, dict) or not patch:
                return {"error": "config_patch must be a non-empty object"}
            self.services[service]["config"].update(patch)
            # Config is written but not live. An agent that skips the restart
            # and submits will fail the healthcheck, which is the lesson.
            return {
                "status": "written",
                "note": f"{service} configuration updated; restart required to take effect",
                "current_config": self.services[service]["config"],
            }

        if name == "restart_service":
            service = kwargs.get("service_name", "")
            if service not in self.services:
                return self._unknown_service(service)
            self.restarted.add(service)
            if self._config_fixed() and service == self.spec["goal"]["service"]:
                self.services[service]["status"] = "healthy"
                self.services[service]["logs"].append("[INFO] restarted; probes healthy")
            else:
                self.services[service]["logs"].append(
                    "[WARN] restarted with unchanged or insufficient configuration"
                )
            return {"status": "restarted", "service": service}

        if name == "run_healthcheck":
            goal = self.spec["goal"]
            ok = self._config_fixed() and goal["service"] in self.restarted
            self.healthcheck_passed = ok
            if ok:
                return {
                    "endpoint": kwargs.get("endpoint", goal["endpoint"]),
                    "status_code": 200,
                    "latency_ms": 18,
                    "message": "OK",
                }
            return {
                "endpoint": kwargs.get("endpoint", goal["endpoint"]),
                "status_code": 503,
                "latency_ms": 3200,
                "message": "still failing: the underlying condition is unchanged",
            }

        if name == "submit_resolution":
            self.submitted = {
                "root_cause": str(kwargs.get("root_cause", "")),
                "action_taken": str(kwargs.get("action_taken", "")),
            }
            return {"status": "submitted"}

        return {
            "error": f"unknown action {name!r}",
            "available": [t["function"]["name"] for t in self.tool_schemas],
        }

    def _unknown_service(self, service: str) -> dict[str, Any]:
        return {"error": f"unknown service {service!r}", "available": sorted(self.services)}

    # -- verification ------------------------------------------------------

    def _config_fixed(self) -> bool:
        """Is the *correct* configuration in place?

        Checks the required key, any secondary keys, and -- for tasks that have
        one -- rejects the plausible-but-wrong fix explicitly. Without that
        last check, ``rate_limit`` would be passed by raising the rate limit,
        which is the behaviour the task exists to catch.
        """
        goal = self.spec["goal"]
        config = self.services.get(goal["service"], {}).get("config", {})
        value = config.get(goal["key"])

        if "min_value" in goal:
            if not isinstance(value, (int, float)) or value < goal["min_value"]:
                return False
        elif "max_value" in goal:
            if not isinstance(value, (int, float)) or value > goal["max_value"]:
                return False
        elif "expected" in goal:
            if value != goal["expected"]:
                return False

        for key, accepted in (goal.get("also") or {}).items():
            if config.get(key) not in accepted:
                return False
        return True

    def _took_wrong_fix(self) -> bool:
        wrong = self.spec["goal"].get("wrong_fix")
        if not wrong:
            return False
        original = TASKS[self.task_id]["services"][wrong["service"]]["config"][wrong["key"]]
        current = self.services.get(wrong["service"], {}).get("config", {}).get(wrong["key"])
        return current != original

    def evaluate(self) -> EvaluationResult:
        goal = self.spec["goal"]
        config_ok = self._config_fixed()
        restarted_ok = goal["service"] in self.restarted
        verified = self.healthcheck_passed
        submitted = self.submitted is not None
        masked = self._took_wrong_fix()

        success = config_ok and restarted_ok and verified and submitted and not masked
        # Partial credit tracks the four-beat discipline, so the curriculum and
        # the optimizer can see an agent getting closer rather than only
        # pass/fail. Masking the symptom forfeits it.
        score = 0.0 if masked else 0.25 * sum([config_ok, restarted_ok, verified, submitted])

        return EvaluationResult(
            success=success,
            score=1.0 if success else score,
            metrics={
                "config_ok": config_ok,
                "restarted_ok": restarted_ok,
                "verified": verified,
                "submitted": submitted,
                "masked_symptom": masked,
                "steps": self.step_count,
            },
        )

    # -- views -------------------------------------------------------------

    def observe(self) -> Observation:
        statuses = {name: entry["status"] for name, entry in self.services.items()}
        lines = [
            f"=== INCIDENT: {self.spec['title']} ===",
            self.spec["instruction"],
            f"Services: {statuses}",
            f"Step {self.step_count}/{self.max_steps}",
        ]
        if self.last_result is not None:
            lines.append(f"Last action returned: {self.last_result}")
        return Observation(
            text="\n".join(lines),
            data={
                "task_id": self.task_id,
                "services": statuses,
                "last_result": self.last_result,
            },
        )

    def get_env_state(self) -> dict[str, Any]:
        # `verified` is the flag VerificationGate reads. Naming it generically
        # is what lets that harness apply to any benchmark.
        return {
            "task_id": self.task_id,
            "services": {n: dict(e["config"]) for n, e in self.services.items()},
            "step_count": self.step_count,
            "restarted": sorted(self.restarted),
            "verified": self.healthcheck_passed,
            "healthcheck_passed": self.healthcheck_passed,
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_benchmark("devops")
class DevOpsBenchmark(BenchmarkAdapter):
    """Five SRE incidents; deterministic, offline, no external services."""

    description = "Production incident diagnosis with verifiable ground truth."

    #: Held back from training so a run reports a number on tasks the memory
    #: bank has never been induced from.
    EVAL_TASKS = ("rate_limit", "disk_pressure")

    def __init__(self, max_steps: int = 20, include_eval_in_train: bool = False) -> None:
        self.max_steps = int(max_steps)
        self.include_eval_in_train = bool(include_eval_in_train)

    def task_ids(self, split: str = "train") -> list[str]:
        every = list(TASKS)
        if split == "all" or self.include_eval_in_train:
            return every
        if split == "eval":
            return [t for t in every if t in self.EVAL_TASKS]
        return [t for t in every if t not in self.EVAL_TASKS]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0):
        return DevOpsIncidentEnv(task_id=task_id, max_steps=self.max_steps)

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=tid,
                instruction=spec["instruction"],
                split="eval" if tid in self.EVAL_TASKS else "train",
                difficulty=float(spec.get("difficulty", 0.5)),
                tags=["sre", "diagnosis"],
            )
            for tid, spec in TASKS.items()
        ]

    def instruction_for(self, task_id: str) -> str:
        return TASKS.get(task_id, {}).get("instruction", task_id)

    def system_prompt(self) -> str:
        return super().system_prompt().replace(
            "You are an autonomous problem-solving agent operating inside an instrumented\nenvironment.",
            "You are an autonomous site-reliability agent responding to a production\nincident.",
        )
