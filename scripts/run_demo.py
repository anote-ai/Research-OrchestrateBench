#!/usr/bin/env python3
"""Demo: compare FixedPolicy vs HeuristicPolicy on 20 tasks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestratebench.core import FixedPolicy, HeuristicPolicy, OrchestratorBench
from orchestratebench.data import make_benchmark_tasks
from orchestratebench.evaluate import policy_comparison, routing_distribution

try:
    from rich.table import Table
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def main() -> None:
    tasks = make_benchmark_tasks(n=20, seed=42)
    bench = OrchestratorBench(tasks=tasks)

    policies = {
        "FixedPolicy": FixedPolicy(),
        "HeuristicPolicy": HeuristicPolicy(),
    }
    all_traces = bench.compare_policies(policies)
    comparison = policy_comparison(all_traces)

    if HAS_RICH:
        console = Console()
        table = Table(title="Policy Comparison")
        table.add_column("Policy")
        table.add_column("Success Rate", justify="right")
        table.add_column("Mean Latency (ms)", justify="right")
        table.add_column("Mean Cost (USD)", justify="right")
        table.add_column("Efficiency", justify="right")
        table.add_column("N Traces", justify="right")
        for policy_name, stats in comparison.items():
            table.add_row(
                policy_name,
                f"{stats['success_rate']:.3f}",
                f"{stats['mean_latency']:.0f}",
                f"{stats['mean_cost']:.4f}",
                f"{stats['orchestration_efficiency_score']:.3f}",
                str(stats["n_traces"]),
            )
        console.print(table)
        for policy_name, traces in all_traces.items():
            dist = routing_distribution(traces)
            console.print(f"[bold]{policy_name}[/bold] routing distribution: {dist}")
    else:
        for policy_name, stats in comparison.items():
            print(f"{policy_name}: {stats}")


if __name__ == "__main__":
    main()
