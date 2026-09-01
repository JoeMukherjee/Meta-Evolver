"""Model access, through LangChain.

Chat models are LangChain ``BaseChatModel`` instances, which is what makes the
rest of this package idiomatic LangGraph rather than LangGraph-shaped: state
carries real ``AnyMessage`` objects under the ``add_messages`` reducer, tool
calls arrive already normalized on ``AIMessage.tool_calls``, and a test double
is just another ``BaseChatModel``. Nothing here re-implements a message format.

Two provider details are handled centrally, because scattering them across
call sites guarantees drift:

**Gemini removed the manual sampling overrides.** ``temperature``, ``top_p``
and ``top_k`` are deprecated on the Gemini API -- generation is steered by the
thinking level instead. :func:`build_chat_model` strips them for any Gemini
route while leaving them intact for providers that still honour them, so a
config carrying ``temperature: 0.4`` stays correct on both.

**Model strings accept either spelling.** ``gemini/gemini-3-flash`` (the
provider-prefixed form this project used before) and
``google_genai:gemini-3-flash`` (LangChain's) both resolve, so existing
configs and CLI invocations keep working.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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

#: Prefixes this project has used, mapped to LangChain provider ids.
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google_genai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "google-genai": "google_genai",
    "vertex_ai": "google_vertexai",
    "google_vertexai": "google_vertexai",
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "mistral": "mistralai",
    "mistralai": "mistralai",
    "ollama": "ollama",
    "together_ai": "together",
    "together": "together",
    "fireworks": "fireworks",
    "openrouter": "openrouter",
}

#: Bare model names, resolved by family prefix.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gemini", "google_genai"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("mistral", "mistralai"),
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

#: Default chat model.
DEFAULT_MODEL = "google_genai:gemini-3-flash"

#: Default embedding model.
#:
#: ``gemini-embedding-2`` over ``gemini-embedding-001`` for one reason that
#: matters at reduced width: it **renormalizes truncated embeddings
#: automatically**. Both are Matryoshka-trained, so a 3072-dim vector can be
#: cut to 768 without meaningful quality loss -- but on ``-001`` the truncated
#: vector is no longer unit-norm, and every consumer has to renormalize it or
#: silently compare by magnitude as well as direction.
DEFAULT_EMBED_MODEL = "google_genai:gemini-embedding-2"

#: Default embedding width: 768 of the model's 3072 available dimensions.
#:
#: Matryoshka Representation Learning packs the most significant structure
#: into the leading dimensions, so the first 768 carry nearly all the
#: retrieval signal at a quarter of the storage and a quarter of the
#: dot-product cost. Both matter here: a memory bank persists every vector to
#: JSONL, and MMR retrieval is O(k*n) dot products per episode.
#:
#: Recommended values are 768, 1536 and 3072; 128-3072 is accepted. ``None``
#: takes the model default (3072).
DEFAULT_EMBED_DIMENSIONS = 768


def split_model(spec: str) -> tuple[str, str]:
    """``spec`` -> ``(langchain_provider, model_name)``.

    Accepts ``provider:model`` (LangChain), ``provider/model`` (the form this
    project used with litellm), and bare names resolved by family prefix.
    An unrecognized provider is passed through untouched so a new LangChain
    integration works before this table knows about it.
    """
    spec = (spec or "").strip()
    if not spec:
        return "", ""

    for sep in (":", "/"):
        if sep in spec:
            head, tail = spec.split(sep, 1)
            provider = _PROVIDER_ALIASES.get(head.lower())
            if provider:
                return provider, tail
            if head.lower() == "models":
                break
            # An unknown head is still a provider hint -- a new LangChain
            # integration should work before this table knows its name.
            return head, tail

    low = spec.lower()
    for prefix, provider in _FAMILY_PREFIXES:
        if low.startswith(prefix):
            return provider, spec
    return "", spec


def qualify(spec: str) -> str:
    """Model id in LangChain's ``provider:model`` form."""
    provider, name = split_model(spec)
    return f"{provider}:{name}" if provider else name


def sampling_params_deprecated(spec: str) -> bool:
    """True when ``spec`` rejects / ignores temperature, top_p and top_k.

    Covers every Gemini route -- the ``gemini/`` and ``google_genai:``
    prefixes, ``vertex_ai/gemini-*``, and bare ``gemini-*`` names -- so a call
    cannot smuggle a deprecated knob through by spelling the model
    differently.
    """
    provider, name = split_model(spec)
    return provider == "google_genai" or "gemini" in (name or "").lower()


def effective_model(spec: str) -> str:
    """``spec``, unless ``$META_EVOLVER_MODEL`` overrides it run-wide.

    One switch reaches every stage -- policy, memory induction, prompt
    optimization -- including stages behind a subprocess that inherits the
    environment but not the command line.
    """
    load_dotenv_once()
    return os.environ.get("META_EVOLVER_MODEL") or spec


class LLMError(RuntimeError):
    """Raised when a model call fails after the retry budget is spent."""


# --- construction -----------------------------------------------------------


def build_rate_limiter(requests_per_second: float | None) -> Any:
    """A client-side limiter, or ``None`` to leave requests unthrottled.

    Rollouts fan out concurrently against one API key, so the natural failure
    is a burst of 429s at the start of every generation. Retries recover from
    that but pay full backoff for it; pacing the requests is cheaper and makes
    a run's wall-clock predictable.

    ``check_every_n_seconds`` is deliberately much finer than the rate: the
    limiter sleeps in those increments, so a coarse value would quantise every
    request onto a slow tick.
    """
    if not requests_per_second or requests_per_second <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=float(requests_per_second),
        check_every_n_seconds=min(0.1, 1.0 / (float(requests_per_second) * 4)),
        # Allow a small burst so a generation's first few rollouts start
        # immediately rather than queuing behind the very first tick.
        max_bucket_size=max(1.0, float(requests_per_second)),
    )


def build_chat_model(
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
    timeout: float | None = 120.0,
    requests_per_second: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """A LangChain chat model for ``model``.

    ``max_retries`` is LangChain's own transient-error retry, which covers
    rate limits, 5xx and connection failures without retrying a deterministic
    400 -- burning a minute of backoff to fail identically helps nobody.

    ``requests_per_second`` throttles client-side. Worth setting whenever
    rollouts run concurrently against a single key.
    """
    from langchain.chat_models import init_chat_model

    load_dotenv_once()
    spec = effective_model(model)
    provider, name = split_model(spec)

    if sampling_params_deprecated(spec):
        for param in DEPRECATED_SAMPLING_PARAMS:
            kwargs.pop(param, None)

    limiter = build_rate_limiter(requests_per_second)
    if limiter is not None:
        kwargs.setdefault("rate_limiter", limiter)

    init_kwargs: dict[str, Any] = {"max_retries": max_retries, **kwargs}
    if timeout is not None:
        init_kwargs.setdefault("timeout", timeout)
    if provider:
        init_kwargs["model_provider"] = provider

    try:
        return init_chat_model(name, **init_kwargs)
    except Exception as exc:
        raise LLMError(
            f"could not build chat model {spec!r} "
            f"(provider={provider or 'inferred'}, model={name!r}): {exc}. "
            "Install the provider package, e.g. `pip install langchain-google-genai`."
        ) from exc


def model_name(model: BaseChatModel) -> str:
    """A readable identifier for a chat model, for logs and reports."""
    for attr in ("model", "model_name", "_llm_type"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def message_text(message: BaseMessage) -> str:
    """Plain text of a message, whether its content is a string or blocks.

    LangChain models may reply with structured content blocks rather than a
    plain string; stringifying those directly would put a list repr into a
    prompt, which is the kind of bug that surfaces as "the optimizer got
    worse" rather than as an error.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def invoke_text(model: BaseChatModel, messages: Sequence[BaseMessage], **kwargs: Any) -> str:
    """Invoke ``model`` and return its text content.

    Wraps provider exceptions in :class:`LLMError` so the graph's error
    handling has one type to catch.
    """
    try:
        response = model.invoke(list(messages), **kwargs)
    except Exception as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    return message_text(response)


# --- accounting -------------------------------------------------------------


class TokenMeter:
    """Total token usage for every model call made inside the block.

    Uses LangChain's usage callback rather than summing ``usage_metadata`` at
    each call site, so it also captures the calls this package makes
    indirectly -- memory induction, prompt proposals, validation rollouts --
    which is exactly the spend a per-generation cost figure should include.

    ::

        with TokenMeter() as meter:
            ...
        print(meter.total)

    ``total`` stays readable after the block closes, which is the whole point:
    the generation that spent the tokens reports them one node later.
    """

    def __init__(self) -> None:
        self._cm: Any = None
        self._callback: Any = None
        self._total = 0

    def __enter__(self) -> TokenMeter:
        from langchain_core.callbacks import get_usage_metadata_callback

        self._cm = get_usage_metadata_callback()
        self._callback = self._cm.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._total = self._read()
        if self._cm is not None:
            self._cm.__exit__(*exc)
            self._cm = None

    def _read(self) -> int:
        usage = getattr(self._callback, "usage_metadata", None) or {}
        return sum(int(counts.get("total_tokens", 0) or 0) for counts in usage.values())

    @property
    def total(self) -> int:
        """Tokens counted so far, or in total once the block has closed."""
        return self._read() if self._cm is not None else self._total

    @property
    def by_model(self) -> dict[str, Any]:
        return dict(getattr(self._callback, "usage_metadata", None) or {})


# --- test double ------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """A deterministic ``BaseChatModel`` for tests and offline demos.

    Takes either a fixed script of ``AIMessage`` replies or a callable policy
    over the message list. Because it is a real chat model, the whole engine
    -- episode graph, evolution graph, memory, curriculum -- runs against it
    end to end with no network, which is how the test suite stays fast.

    ``bind_tools`` records what it was offered and returns ``self``, so a test
    can assert on the tool set the agent was actually shown.
    """

    responder: Any = None
    script: list[Any] = []
    calls: list[Any] = []
    bound_tools: list[Any] | None = None
    cursor: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        script: Sequence[AIMessage] | None = None,
        responder: Callable[[list[BaseMessage], list[Any] | None], AIMessage] | None = None,
        **kwargs: Any,
    ) -> None:
        if script is not None and responder is not None:
            raise ValueError("pass either script or responder, not both")
        super().__init__(**kwargs)
        self.script = list(script or [])
        self.responder = responder
        self.calls = []
        self.bound_tools = None
        self.cursor = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.responder is not None:
            reply = self.responder(list(messages), self.bound_tools)
        elif self.cursor < len(self.script):
            reply = self.script[self.cursor]
            self.cursor += 1
        else:
            reply = AIMessage(content="(end of script)")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call_message(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    """An ``AIMessage`` carrying one tool call. Convenience for tests."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
