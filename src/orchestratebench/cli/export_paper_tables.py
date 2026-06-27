"""Export publication-friendly markdown and LaTeX from summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from orchestratebench.paper_reports import (
    build_exp2_latex_tables,
    build_exp2_markdown_report,
    build_exp3_latex_tables,
    build_exp3_markdown_report,
    write_publication_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["2", "3", "exp2", "exp3"],
        help="Experiment artifact type.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to the experiment summary.json artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for paper_summary.md and paper_tables.tex. Defaults to summary.json parent.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = "2" if args.experiment in {"2", "exp2"} else "3"
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.summary_json.parent

    if experiment == "2":
        markdown_text = build_exp2_markdown_report(
            payload["summary_by_mode"],
            payload["pairwise_by_mode"],
            config=payload.get("config"),
        )
        latex_text = build_exp2_latex_tables(payload["summary_by_mode"])
    else:
        markdown_text = build_exp3_markdown_report(
            payload["summary_by_depth_stage"],
            payload["summary_by_depth"],
            payload["pairwise_by_depth_stage"],
            payload["pairwise_by_depth"],
            config=payload.get("config"),
        )
        latex_text = build_exp3_latex_tables(
            payload["summary_by_depth_stage"],
            payload["summary_by_depth"],
        )

    write_publication_artifacts(
        output_dir,
        markdown_text=markdown_text,
        latex_text=latex_text,
    )
    print(f"[artifact] wrote publication-friendly outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
