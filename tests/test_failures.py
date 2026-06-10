"""Tests for orchestratebench.failures."""

from __future__ import annotations

from orchestratebench.core import (
    ExecutionTrace,
    FixedPolicy,
    HeuristicPolicy,
    OrchestratorAction,
    RoutingDecision,
    SubAgentType,
    TaskStatus,
)
from orchestratebench.data import (
    make_benchmark_tasks,
    make_execution_trace,
    make_finance_approval_workflow,
    make_hr_onboarding_workflow,
)
from orchestratebench.failures import (
    FailureInjector,
    FailureMode,
    measure_cascade,
    recovery_rate_by_mode,
)

import pytest


def _make_trace(success: bool = True) -> ExecutionTrace:
    tasks = make_benchmark_tasks(n=1, seed=0)
    return make_execution_trace(tasks[0], success=success)


# ---------------------------------------------------------------------------
# FailureInjector — each mode produces a failed trace
# ---------------------------------------------------------------------------


class TestFailureInjector:
    def test_ambiguous_delegation_fails_trace(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        corrupted = injector.inject(original, FailureMode.AMBIGUOUS_DELEGATION)
        assert not corrupted.success
        assert corrupted.status == TaskStatus.FAILED
        assert corrupted.actions[0].confidence < 0.6

    def test_tool_invocation_error_adds_retry(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        corrupted = injector.inject(original, FailureMode.TOOL_INVOCATION_ERROR)
        assert not corrupted.success
        assert len(corrupted.retries) == len(original.retries) + 1
        assert "ToolInvocationError" in corrupted.retries[-1].error

    def test_context_pollution_inflates_cost(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        corrupted = injector.inject(original, FailureMode.CONTEXT_POLLUTION)
        assert not corrupted.success
        assert corrupted.total_cost_usd >= original.total_cost_usd

    def test_conflicting_outputs_adds_extra_action(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        corrupted = injector.inject(original, FailureMode.CONFLICTING_OUTPUTS)
        assert not corrupted.success
        assert len(corrupted.actions) == len(original.actions) + 1

    def test_premature_action_changes_to_reason_only(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        corrupted = injector.inject(original, FailureMode.PREMATURE_ACTION)
        assert not corrupted.success
        assert corrupted.actions[0].decision == RoutingDecision.REASON_ONLY

    def test_deterministic_with_seed(self) -> None:
        t = _make_trace()
        a = FailureInjector(seed=42).inject(t, FailureMode.AMBIGUOUS_DELEGATION)
        b = FailureInjector(seed=42).inject(t, FailureMode.AMBIGUOUS_DELEGATION)
        assert a == b

    def test_unknown_mode_raises(self) -> None:
        injector = FailureInjector()
        with pytest.raises(ValueError, match="Unknown failure mode"):
            injector.inject(_make_trace(), "nonexistent_mode")  # type: ignore[arg-type]

    def test_original_trace_unmodified(self) -> None:
        injector = FailureInjector(seed=0)
        original = _make_trace(success=True)
        original_success = original.success
        _ = injector.inject(original, FailureMode.TOOL_INVOCATION_ERROR)
        assert original.success == original_success


# ---------------------------------------------------------------------------
# measure_cascade — cascade propagation (issue #7)
# ---------------------------------------------------------------------------


class TestMeasureCascade:
    def test_injection_at_stage_0_cascades_to_dependents(self) -> None:
        tasks = make_finance_approval_workflow()
        result = measure_cascade(
            tasks=tasks,
            policy=HeuristicPolicy(),
            injection_stage=0,
            failure_mode=FailureMode.TOOL_INVOCATION_ERROR,
            seed=0,
        )
        assert result["cascade_radius"] >= 1
        assert result["total_tasks"] == 4
        assert 0.0 <= result["recovery_completeness"] <= 1.0

    def test_injection_at_last_stage_has_zero_cascade(self) -> None:
        tasks = make_finance_approval_workflow()
        result = measure_cascade(
            tasks=tasks,
            policy=HeuristicPolicy(),
            injection_stage=len(tasks) - 1,
            failure_mode=FailureMode.AMBIGUOUS_DELEGATION,
            seed=0,
        )
        assert result["cascade_radius"] == 0

    def test_no_injection_all_succeed(self) -> None:
        tasks = make_hr_onboarding_workflow()
        result = measure_cascade(
            tasks=tasks,
            policy=FixedPolicy(),
            injection_stage=999,
            failure_mode=FailureMode.TOOL_INVOCATION_ERROR,
            seed=0,
        )
        injection_idx = result["injection_stage"]
        non_injected = [
            t for i, t in enumerate(result["traces"]) if i != injection_idx
        ]
        assert all(t.success for t in non_injected)

    def test_result_contains_all_traces(self) -> None:
        tasks = make_finance_approval_workflow()
        result = measure_cascade(
            tasks=tasks,
            policy=FixedPolicy(),
            injection_stage=1,
            failure_mode=FailureMode.CONTEXT_POLLUTION,
            seed=0,
        )
        assert len(result["traces"]) == len(tasks)

    def test_deeper_pipeline_wider_cascade(self) -> None:
        short = make_finance_approval_workflow()  # 4 stages
        long = make_hr_onboarding_workflow()  # 5 stages
        r_short = measure_cascade(
            short, HeuristicPolicy(), 0, FailureMode.TOOL_INVOCATION_ERROR, seed=0
        )
        r_long = measure_cascade(
            long, HeuristicPolicy(), 0, FailureMode.TOOL_INVOCATION_ERROR, seed=0
        )
        assert r_long["cascade_radius"] >= r_short["cascade_radius"]


# ---------------------------------------------------------------------------
# recovery_rate_by_mode — aggregate metrics per failure mode
# ---------------------------------------------------------------------------


class TestRecoveryRateByMode:
    def test_returns_all_modes(self) -> None:
        tasks = make_finance_approval_workflow()
        results = recovery_rate_by_mode(tasks, HeuristicPolicy(), n_runs=3, seed=0)
        for mode in FailureMode:
            assert mode.value in results
            assert "recovery_rate" in results[mode.value]
            assert "mean_cascade_radius" in results[mode.value]

    def test_subset_of_modes(self) -> None:
        tasks = make_finance_approval_workflow()
        modes = [FailureMode.TOOL_INVOCATION_ERROR, FailureMode.CONTEXT_POLLUTION]
        results = recovery_rate_by_mode(tasks, FixedPolicy(), modes=modes, n_runs=2, seed=0)
        assert set(results.keys()) == {m.value for m in modes}

    def test_deterministic_with_seed(self) -> None:
        tasks = make_finance_approval_workflow()
        a = recovery_rate_by_mode(tasks, FixedPolicy(), n_runs=5, seed=42)
        b = recovery_rate_by_mode(tasks, FixedPolicy(), n_runs=5, seed=42)
        assert a == b
