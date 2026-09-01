"""Provider-agnostic LLM access.

One chokepoint, for one reason: request parameters have provider-specific
validity, and scattering that knowledge across call sites guarantees drift.
Everything that talks to a model in this package goes through ``LLMClient``.

The rule that motivated the design: **Google removed the manual sampling
overrides from the Gemini API.** ``temperature``, ``top_p`` and ``top_k`` are
deprecated there -- generation is steered by the thinking level instead. Rather
than ask every call site to remember that, ``_prepare`` strips them for any
Gemini route while leaving them intact for providers that still honour them.
A config may keep ``temperature: 0.4`` and stay correct on both.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- environment ------------------------------------------------------------

_DOTENV_LOADED = False


def load_dotenv_once(start: Path | None = None) -> None:
    """Populate ``os.environ`` from the nearest ``.env``, without clobbering.

    Deliberately tiny and dependency-free: an agent framework that cannot be
    imported because ``python-dotenv`` is missing is a bad trade.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents][:4]:
        env_file = candidate / ".env"
        if not env_file.is_file():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError:
            pass
        return


# --- model routing ----------------------------------------------------------

_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "together_ai": ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
}

_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gemini", "gemini"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("claude", "anthropic"),
    ("mistral", "mistral"),
    ("llama", "groq"),
)

#: Sampling knobs the Gemini API no longer accepts.
DEPRECATED_SAMPLING_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "topP",
    "topK",
)


def split_model(model: str) -> tuple[str, str]:
    """``"gemini/gemini-3-flash"`` -> ``("gemini", "gemini-3-flash")``."""
    model = (model or "").strip()
    if "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    low = model.lower()
    for prefix, provider in _FAMILY_PREFIXES:
        if low.startswith(prefix):
            return provider, model
    return "", model


def sampling_params_deprecated(model: str) -> bool:
    """True when ``model`` rejects / ignores temperature, top_p and top_k.

    Covers every Gemini route -- the ``gemini/`` AI Studio prefix,
    ``vertex_ai/gemini-*``, and bare ``gemini-*`` names -- so a call cannot
    smuggle a deprecated knob through by spelling the model differently.
    """
    provider, name = split_model(model)
    return provider == "gemini" or "gemini" in (name or "").lower()


def api_key_for(model: str) -> str | None:
    load_dotenv_once()
    provider, _ = split_model(model)
    for var in _API_KEY_ENV.get(provider, ()):
        if os.environ.get(var):
            return os.environ[var]
    return None


def effective_model(model: str) -> str:
    """``model``, unless ``$META_EVOLVER_MODEL`` overrides it run-wide.

    One switch reaches every stage -- policy, memory induction, prompt
    optimization -- including stages behind a subprocess that inherits the
    environment but not the command line.
    """
    load_dotenv_once()
    return os.environ.get("META_EVOLVER_MODEL") or model


# --- responses --------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """Raised when a call fails after the retry budget is spent."""


# --- clients ----------------------------------------------------------------


class BaseLLMClient:
    """Interface every client honours. Tests substitute ``ScriptedLLMClient``."""

    model: str = ""

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return one vector per text, or ``None`` when unavailable so the
        caller can fall back to a local encoder."""
        return None


class LiteLLMClient(BaseLLMClient):
    """litellm-backed client: any provider litellm supports.

    Retries transient failures (rate limit, 5xx, connection, timeout) with
    exponential backoff. Deterministic 4xx errors are *not* retried -- burning
    ninety seconds of backoff to fail identically helps nobody.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-3-flash",
        embed_model: str = "gemini/gemini-embedding-001",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        max_retries: int = 5,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        **defaults: Any,
    ) -> None:
        self.model = effective_model(model)
        self.embed_model = embed_model
        self.api_key = api_key or api_key_for(self.model)
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = int(max_retries)
        self.retry_initial_delay = float(retry_initial_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.defaults = defaults
        self.n_calls = 0
        self.n_tokens = 0

    # -- request construction ---------------------------------------------

    def _prepare(self, **overrides: Any) -> dict[str, Any]:
        """Build request kwargs the target provider will actually accept."""
        kwargs: dict[str, Any] = {**self.defaults, **overrides}
        kwargs["model"] = self.model
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        kwargs.setdefault("max_tokens", self.max_tokens)
        if self.temperature is not None:
            kwargs.setdefault("temperature", self.temperature)

        if sampling_params_deprecated(self.model):
            # Gemini: the sampling overrides are gone. Drop rather than send.
            for param in DEPRECATED_SAMPLING_PARAMS:
                kwargs.pop(param, None)

        return {k: v for k, v in kwargs.items() if v is not None}

    # -- calls -------------------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm

        litellm.suppress_debug_info = True
        litellm.drop_params = True

        req = self._prepare(max_tokens=max_tokens or self.max_tokens, **kwargs)
        req["messages"] = list(messages)
        if tools:
            req["tools"] = list(tools)
            req.setdefault("tool_choice", "auto")
        if response_format:
            req["response_format"] = response_format

        transient = _transient_exception_types(litellm)
        started = time.time()
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                raw = litellm.completion(**req)
                break
            except transient as exc:  # noqa: PERF203 - retry is the point
                last = exc
                if attempt == self.max_retries:
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                delay = min(self.retry_max_delay, self.retry_initial_delay * 2**attempt)
                time.sleep(delay)
            except Exception as exc:
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMError(str(last))

        return self._parse(raw, (time.time() - started) * 1000.0)

    def _parse(self, raw: Any, latency_ms: float) -> LLMResponse:
        message = raw.choices[0].message
        calls: list[ToolCall] = []
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": args}
            calls.append(
                ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=args or {})
            )

        tokens = 0
        usage = getattr(raw, "usage", None)
        if usage is not None:
            tokens = int(getattr(usage, "total_tokens", 0) or 0)

        self.n_calls += 1
        self.n_tokens += tokens
        return LLMResponse(
            content=message.content or "",
            tool_calls=calls,
            tokens=tokens,
            latency_ms=latency_ms,
            raw=raw,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        import litellm

        try:
            kwargs: dict[str, Any] = {"model": self.embed_model, "input": list(texts)}
            key = api_key_for(self.embed_model)
            if key:
                kwargs["api_key"] = key
            resp = litellm.embedding(**kwargs)
            return [row["embedding"] for row in resp.data]
        except Exception:
            # Callers fall back to the deterministic local encoder. An
            # embedding outage should degrade retrieval quality, not stop a run.
            return None


def _transient_exception_types(litellm_module: Any) -> tuple[type[BaseException], ...]:
    """Exception classes worth retrying, resolved dynamically.

    litellm's module layout has shifted across releases; naming them by string
    keeps this working across versions. ``APIError`` (the base class) is
    excluded on purpose -- it covers deterministic 400s too.
    """
    out: list[type[BaseException]] = []
    for name in (
        "APIConnectionError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "Timeout",
    ):
        cls = getattr(litellm_module, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            out.append(cls)
    return tuple(out) or (ConnectionError,)


class ScriptedLLMClient(BaseLLMClient):
    """Deterministic client for tests and offline demos.

    Accepts either a fixed script of responses or a callable policy over the
    message list. The whole engine -- episode graph, evolution graph, memory,
    curriculum -- runs end to end against this, which is how the test suite
    stays fast and network-free.
    """

    def __init__(
        self,
        script: Iterable[LLMResponse] | None = None,
        responder: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse]
        | None = None,
        model: str = "scripted/deterministic",
    ) -> None:
        if script is not None and responder is not None:
            raise ValueError("pass either script or responder, not both")
        self.model = model
        self.script = list(script or [])
        self.responder = responder
        self._idx = 0
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools=None, response_format=None, max_tokens=None, **kwargs):
        self.calls.append(list(messages))
        if self.responder is not None:
            return self.responder(list(messages), list(tools) if tools else None)
        if self._idx >= len(self.script):
            return LLMResponse(content="(end of script)")
        resp = self.script[self._idx]
        self._idx += 1
        return resp


def build_client(model: str | None = None, **kwargs: Any) -> BaseLLMClient:
    """Default client factory used by the CLI and the graphs."""
    return LiteLLMClient(model=model or "gemini/gemini-3-flash", **kwargs)
