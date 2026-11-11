"""ChatBenchmark -- an open-ended conversation, run through the same engine.

Every other adapter in this package has a fixed task list and a verifier.
A chatbot has neither: there is no ``task_ids()`` to sweep and no ground
truth to score against. What it *does* share with a benchmark task is
everything the engine actually depends on -- an :class:`ActionableEnv` with
tool schemas the model binds against, one episode graph, one memory bank,
one evolving prompt.

The fit: a live conversation is a benchmark with exactly one task ("session")
and no verifier, where the environment's ``step`` blocks on a real human
turn instead of returning a scripted observation. That is the entire
adapter -- no engine change, no parallel code path. ``MetaEvolver.evolve()``
and ``.evaluate()`` still work against it (useful for offline learning from
recorded transcripts); the addition here is a third entry point,
``MetaEvolver.chat()``, for running one *live* session interactively.

::

    from meta_evolver.benchmarks.chat import ChatBenchmark, StdinTurnSource
    from meta_evolver.core.evolver import MetaEvolver

    evolver = MetaEvolver(benchmark=ChatBenchmark(tools={"search": my_search}))
    evolver.chat()  # blocks on the terminal until the user ends the session

A custom frontend (a websocket handler, a Slack bot) supplies its own
:class:`ChatTurnSource` instead of :class:`StdinTurnSource` -- that is the
one seam a new frontend has to fill, exactly as :class:`BenchmarkAdapter` is
the one seam a new evaluation suite has to fill.
"""
from __future__ import annotations

import json
import queue
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.benchmarks.custom import derive_schema
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

#: Tool name the agent calls to speak. Reserved -- a custom tool with this
#: name would be shadowed and never reachable.
SEND_MESSAGE = "send_message"

#: Tool name the agent calls to end the session on its own initiative.
END_CONVERSATION = "end_conversation"


# ---------------------------------------------------------------------------
# Turn sources -- where the next human message comes from
# ---------------------------------------------------------------------------


class ChatTurnSource(ABC):
    """Supplies the human side of the conversation, one message at a time.

    The one seam a new frontend fills. A CLI polls stdin
    (:class:`StdinTurnSource`); a server hands messages off a queue as they
    arrive over a websocket (:class:`QueueTurnSource`); a test scripts a
    fixed list (:class:`ScriptedTurnSource`).
    """

    @abstractmethod
    def next_message(self) -> str | None:
        """The next human message, or ``None`` when the human has left."""


class StdinTurnSource(ChatTurnSource):
    """Reads from the terminal. Typing ``exit``, ``quit`` or ``/end`` leaves."""

    END_WORDS = {"exit", "quit", "/end", "/quit"}

    def __init__(self, prompt: str = "You: ") -> None:
        self.prompt = prompt

    def next_message(self) -> str | None:
        try:
            text = input(self.prompt)
        except EOFError:
            return None
        if text.strip().lower() in self.END_WORDS:
            return None
        return text


class QueueTurnSource(ChatTurnSource):
    """Backed by a thread-safe queue -- for a server, a UI, or a test.

    A frontend calls :meth:`send` from whatever thread receives the user's
    message (a websocket handler, a request thread); ``next_message`` blocks
    the episode until one arrives. Call :meth:`end` to close the session
    without a further message.
    """

    def __init__(self, timeout: float | None = None) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self.timeout = timeout

    def send(self, text: str) -> None:
        self._q.put(text)

    def end(self) -> None:
        self._q.put(None)

    def next_message(self) -> str | None:
        try:
            return self._q.get(timeout=self.timeout)
        except queue.Empty:
            return None


class ScriptedTurnSource(ChatTurnSource):
    """A fixed list of human turns, for tests and offline demos."""

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self._i = 0

    def next_message(self) -> str | None:
        if self._i >= len(self._turns):
            return None
        turn = self._turns[self._i]
        self._i += 1
        return turn


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


class LiveChatEnv(ActionableEnv):
    """One open-ended conversation, driven by a real :class:`ChatTurnSource`.

    There is no fixed task and no verifier. The agent's reply to the user
    *is* its action (``send_message``); the environment's response to that
    action is the user's next turn, fetched from ``turn_source``. The
    conversation ends when the turn source runs out (the human left) or the
    agent calls ``end_conversation``. ``max_steps`` on the episode -- an
    ordinary :func:`~meta_evolver.graphs.episode.arun_episode` argument, not
    anything new here -- is the safety cap against a runaway loop.

    Blocking on ``turn_source.next_message()`` inside ``step`` holds the
    event loop for as long as the human takes to reply, same as the docstring
    on :mod:`meta_evolver.graphs.episode` warns any real-I/O environment
    will. That is the intended behaviour for one interactive session; a
    frontend that runs several chat episodes concurrently in one process
    should give each its own thread rather than share an event loop.
    """

    env_type = "live_chat"

    def __init__(
        self,
        turn_source: ChatTurnSource,
        system_intro: str = "",
        extra_tools: dict[str, Callable[..., Any]] | None = None,
        on_reply: Callable[[str], None] | None = None,
    ) -> None:
        self.turn_source = turn_source
        self.system_intro = system_intro
        self.extra_tools = dict(extra_tools or {})
        self.on_reply = on_reply or (lambda _text: None)
        self._extra_schemas = [derive_schema(n, f) for n, f in self.extra_tools.items()]
        self.transcript: list[tuple[str, str]] = []
        self.ended = False
        self.n_turns = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvResetResponse:
        self.transcript = []
        self.ended = False
        self.n_turns = 0
        first = self.turn_source.next_message()
        if first is None:
            self.ended = True
            first = "(the user ended the session before saying anything)"
        else:
            self.transcript.append(("user", first))
        return EnvResetResponse(
            observation=Observation(text=first, data={}),
            info={
                "instruction": self.system_intro or "Have a helpful, open-ended conversation.",
                "title": "chat session",
            },
        )

    def step(self, action: Action) -> EnvResponse:
        name, kwargs = action.name, dict(action.kwargs or {})

        if name == END_CONVERSATION:
            self.ended = True
            reason = str(kwargs.get("reason", ""))
            return EnvResponse(
                observation=Observation(text="(conversation ended)"),
                terminated=True,
                info={"result": {"ended_by": "agent", "reason": reason}},
            )

        if name == SEND_MESSAGE:
            text = str(kwargs.get("text", ""))
            self.transcript.append(("assistant", text))
            self.n_turns += 1
            self.on_reply(text)

            nxt = self.turn_source.next_message()
            if nxt is None:
                self.ended = True
                return EnvResponse(
                    observation=Observation(text="(the user left the conversation)"),
                    terminated=True,
                    info={"result": {"ended_by": "user"}},
                )
            self.transcript.append(("user", nxt))
            return EnvResponse(observation=Observation(text=nxt), terminated=False)

        fn = self.extra_tools.get(name)
        if fn is None:
            result: dict[str, Any] = {
                "error": f"unknown tool {name!r}",
                "available": sorted([SEND_MESSAGE, END_CONVERSATION, *self.extra_tools]),
            }
        else:
            try:
                raw = fn(**kwargs)
                result = raw if isinstance(raw, dict) else {"value": raw}
            except Exception as exc:  # a tool error is information, not a crash
                result = {"error": f"{type(exc).__name__}: {exc}"}
        return EnvResponse(
            observation=Observation(text=json.dumps(result, default=str)),
            terminated=False,
            info={"result": result},
        )

    def observe(self) -> Observation:
        last = self.transcript[-1][1] if self.transcript else ""
        return Observation(text=last)

    def evaluate(self) -> EvaluationResult:
        """No ground truth exists for an open-ended conversation.

        ``success``/``score`` here mean only "the session ran to a clean
        end", not "the agent did well" -- real signal (a thumbs-up, an
        explicit correction, whether the user kept talking) has to come from
        the frontend, which is why this stays a stub rather than guessing.
        """
        return EvaluationResult(
            success=True, score=1.0, metrics={"n_turns": self.n_turns, "ended": self.ended}
        )

    def available_tools(self) -> list[dict[str, Any]]:
        return [
            tool_schema(
                SEND_MESSAGE,
                "Send a reply to the user. This is how you speak -- call it "
                "whenever you have something to say.",
                {"text": {"type": "string", "description": "What to say to the user."}},
                required=["text"],
            ),
            tool_schema(
                END_CONVERSATION,
                "End the conversation on your own initiative (rare -- normally "
                "the user ends it).",
                {"reason": {"type": "string", "description": "Why the conversation is ending."}},
            ),
            *self._extra_schemas,
        ]

    def get_env_state(self) -> dict[str, Any]:
        return {"n_turns": self.n_turns, "ended": self.ended}

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


@register_benchmark("chat")
class ChatBenchmark(BenchmarkAdapter):
    """Wraps :class:`LiveChatEnv` as a :class:`BenchmarkAdapter`.

    ``make_env`` needs a live ``turn_source`` per session, which
    ``BenchmarkAdapter.make_env``'s signature has no room for (adding a
    required parameter there would break every other adapter's calling
    convention in ``evolution.py``/``evolver.py``). Instead ``turn_source``
    and ``on_reply`` are set as attributes immediately before a session
    starts -- :meth:`~meta_evolver.core.evolver.MetaEvolver.chat` does this,
    so direct use of this adapter should go through it rather than calling
    ``make_env`` cold.
    """

    def __init__(
        self,
        name: str = "chat",
        description: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Open-ended chat session {name!r}."
        self.tools = dict(tools or {})
        self._system_prompt = system_prompt
        #: Set by ``MetaEvolver.chat()``/``achat()`` before ``make_env`` runs.
        self.turn_source: ChatTurnSource | None = None
        self.on_reply: Callable[[str], None] | None = None

    def task_ids(self, split: str = "train") -> list[str]:
        return ["session"]

    def make_env(
        self, task_id: str, curriculum_level: float = 0.0, seed: int = 0
    ) -> LiveChatEnv:
        if self.turn_source is None:
            raise RuntimeError(
                "ChatBenchmark has no turn_source set. Use "
                "MetaEvolver.chat()/achat(), which sets one per session, "
                "rather than calling make_env() directly."
            )
        return LiveChatEnv(
            turn_source=self.turn_source,
            system_intro=self._system_prompt or "",
            extra_tools=self.tools,
            on_reply=self.on_reply,
        )

    def instruction_for(self, task_id: str) -> str:
        return self._system_prompt or "Have a helpful, open-ended conversation with the user."

    def specs(self) -> list[TaskSpec]:
        return [TaskSpec(task_id="session", instruction=self.instruction_for("session"))]

    def system_prompt(self) -> str:
        return self._system_prompt or super().system_prompt()


__all__ = [
    "END_CONVERSATION",
    "SEND_MESSAGE",
    "ChatBenchmark",
    "ChatTurnSource",
    "LiveChatEnv",
    "QueueTurnSource",
    "ScriptedTurnSource",
    "StdinTurnSource",
]
