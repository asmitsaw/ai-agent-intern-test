"""
evaluation/run_eval.py — Main evaluation runner.

Usage:
    python evaluation/run_eval.py                    # run all cases
    python evaluation/run_eval.py --category privacy # filter by category
    python evaluation/run_eval.py --verbose          # show full responses

Exit code: 0 if all cases pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from agent.orchestrator import AgentSession
from evaluation.assertions import run_assertions

console = Console()

EVAL_DIR = Path(__file__).parent
VISIBLE_CASES_PATH = EVAL_DIR / "visible-cases.json"
CUSTOM_CASES_PATH = EVAL_DIR / "custom-cases.json"


# ── Case loading ──────────────────────────────────────────────────────────────

def load_cases(category_filter: str | None = None) -> list[dict]:
    cases: list[dict] = []

    for path in (VISIBLE_CASES_PATH, CUSTOM_CASES_PATH):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cases.extend(data.get("cases", []))
        elif isinstance(data, list):
            cases.extend(data)

    if category_filter:
        cases = [c for c in cases if c.get("category") == category_filter]

    return cases


# ── Single-case runner ────────────────────────────────────────────────────────

def run_case(case: dict, verbose: bool = False) -> dict:
    """
    Run a single evaluation case and return a result dict.
    """
    session = AgentSession()
    messages = case.get("messages", [])
    expect = case.get("expect", {})

    last_response = None
    for msg in messages:
        if msg["role"] == "user":
            last_response = session.chat(msg["content"])

    if last_response is None:
        return {
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "passed": False,
            "assertions": [("no_response", False, "No user messages in case")],
            "response_text": "",
        }

    assertion_results = run_assertions(last_response, expect)
    all_passed = all(p for _, p, _ in assertion_results)

    if verbose:
        console.print(f"\n[dim]Response for {case['id']}:[/dim]")
        console.print(f"[dim]{last_response.text[:400]}...[/dim]" if len(last_response.text) > 400 else f"[dim]{last_response.text}[/dim]")

    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "passed": all_passed,
        "assertions": assertion_results,
        "response_text": last_response.text,
    }


# ── Report printing ───────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    console.print()

    # ── Per-case table ────────────────────────────────────────────────────────
    table = Table(title="Evaluation Results — Per Case", box=box.ROUNDED, show_lines=True)
    table.add_column("Case ID", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Failed Assertions")

    for r in results:
        status_str = "[bold green]PASS[/bold green]" if r["passed"] else "[bold red]FAIL[/bold red]"
        failed_assertions = [
            f"{name}: {reason}"
            for name, passed, reason in r["assertions"]
            if not passed
        ]
        failed_str = "\n".join(failed_assertions) if failed_assertions else ""
        table.add_row(r["id"], r["category"], status_str, failed_str)

    console.print(table)

    # ── Category summary ──────────────────────────────────────────────────────
    cat_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        if r["passed"]:
            cat_stats[cat]["passed"] += 1

    cat_table = Table(title="Evaluation Results — By Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan bold")
    cat_table.add_column("Passed", justify="right")
    cat_table.add_column("Total", justify="right")
    cat_table.add_column("Score", justify="right")

    for cat, stats in sorted(cat_stats.items()):
        score = stats["passed"] / stats["total"] * 100
        color = "green" if score == 100 else "yellow" if score >= 50 else "red"
        cat_table.add_row(
            cat,
            str(stats["passed"]),
            str(stats["total"]),
            f"[{color}]{score:.0f}%[/{color}]",
        )

    console.print()
    console.print(cat_table)

    # ── Overall summary ───────────────────────────────────────────────────────
    pct = passed / total * 100 if total else 0
    color = "green" if pct == 100 else "yellow" if pct >= 70 else "red"
    console.print()
    console.print(
        f"[bold]Overall: [{color}]{passed}/{total} passed ({pct:.1f}%)[/{color}][/bold]"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row Evaluation Suite")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--verbose", action="store_true", help="Show agent responses")
    parser.add_argument(
        "--case", default=None, help="Run a single case by ID"
    )
    args = parser.parse_args()

    cases = load_cases(category_filter=args.category)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            console.print(f"[red]Case '{args.case}' not found.[/red]")
            return 1

    if not cases:
        console.print("[yellow]No cases to run.[/yellow]")
        return 0

    console.print(f"\n[bold]Running {len(cases)} evaluation case(s)...[/bold]\n")

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        console.print(f"[dim]({i}/{len(cases)}) {case['id']}...[/dim]", end=" ")
        t0 = time.time()
        result = run_case(case, verbose=args.verbose)
        elapsed = time.time() - t0
        status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
        console.print(f"{status} [dim]({elapsed:.1f}s)[/dim]")
        results.append(result)

    print_results(results)

    all_passed = all(r["passed"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
