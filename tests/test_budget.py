"""The spend ledger and the guard that makes it a hard stop."""
import asyncio
import json
import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from meta_evolver.llm.budget import (
    LEDGER_ENV,
    BudgetExceeded,
    BudgetGuard,
    SpendLedger,
    default_ledger,
)


def result(total_tokens: int) -> LLMResult:
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": total_tokens})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def ledger(tmp_path, **kw):
    return SpendLedger(tmp_path / "spend.json", cap_usd=10.0, usd_per_mtok=1.0, reserve_usd=1.0, **kw)


def test_cost_is_tokens_times_the_blended_rate(tmp_path):
    led = ledger(tmp_path)
    led.add(2_500_000)
    assert led.spent_usd == pytest.approx(2.5) and led.remaining_usd == pytest.approx(7.5)
    assert led.remaining_tokens == 7_500_000


def test_ledger_persists_across_processes(tmp_path):
    led = ledger(tmp_path)
    led.add(3_000_000, note="locomo smoke")
    again = SpendLedger(tmp_path / "spend.json")
    assert again.tokens == 3_000_000 and again.cap_usd == 10.0  # cap and rate come from the file
    assert json.loads((tmp_path / "spend.json").read_text())["entries"][0]["note"] == "locomo smoke"


def test_guard_records_completed_calls(tmp_path):
    led = ledger(tmp_path)
    BudgetGuard(led).on_llm_end(result(1234))
    assert led.tokens == 1234


def test_guard_refuses_the_next_call_once_cap_minus_reserve_is_reached(tmp_path):
    led = ledger(tmp_path)
    guard = BudgetGuard(led)
    led.add(8_900_000)  # $8.90 of a $10 cap with a $1 reserve: still allowed
    guard.on_chat_model_start({}, [[]])
    led.add(200_000)  # $9.10 >= $9.00: refuse
    with pytest.raises(BudgetExceeded):
        guard.on_chat_model_start({}, [[]])
    with pytest.raises(BudgetExceeded):
        guard.on_llm_start({}, ["p"])


def test_budget_exceeded_cannot_be_swallowed_by_a_generic_except():
    """The whole point: the engine turns exceptions into per-episode errors and
    carries on, so the guard must not be an ``Exception``."""
    assert not issubclass(BudgetExceeded, Exception) and issubclass(BudgetExceeded, BaseException)

    def episode():
        try:
            raise BudgetExceeded("over")
        except Exception:  # what the engine's infrastructure-error handler does
            return "swallowed"

    with pytest.raises(BudgetExceeded):
        episode()


def test_budget_exceeded_ends_an_asyncio_run():
    async def main():
        await asyncio.gather(asyncio.sleep(0), _boom())

    async def _boom():
        raise BudgetExceeded("over")

    with pytest.raises(BudgetExceeded):
        asyncio.run(main())


def test_concurrent_calls_never_lose_updates(tmp_path):
    led = ledger(tmp_path)
    guard = BudgetGuard(led)
    threads = [threading.Thread(target=lambda: [guard.on_llm_end(result(10)) for _ in range(50)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert led.tokens == 8 * 50 * 10
    assert SpendLedger(tmp_path / "spend.json").tokens == led.tokens


def test_default_ledger_is_absent_without_a_file_and_found_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv(LEDGER_ENV, str(tmp_path / "none.json"))
    assert default_ledger() is None  # a fresh checkout is not silently capped
    p = tmp_path / "real.json"
    SpendLedger(p, cap_usd=5.0).save()
    monkeypatch.setenv(LEDGER_ENV, str(p))
    assert default_ledger().cap_usd == 5.0


def test_built_models_carry_the_guard_when_a_ledger_exists(tmp_path, monkeypatch):
    p = tmp_path / "real.json"
    SpendLedger(p).save()
    monkeypatch.setenv(LEDGER_ENV, str(p))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    from meta_evolver.llm.client import build_chat_model

    model = build_chat_model("google_genai:gemini-3.8-flash")
    assert any(isinstance(c, BudgetGuard) for c in (model.callbacks or []))


def test_summary_is_human_readable(tmp_path):
    led = ledger(tmp_path)
    led.add(4_000_000)
    assert "spent $4.00 of $10.00" in led.summary() and "$6.00 left" in led.summary()
