"""
scripts/run_agent.py — Interactive CLI for the Aster & Row support agent.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --debug       # enable structured logging
    python scripts/run_agent.py --session-id my_session

Commands during chat:
    /reset   — Clear conversation history
    /exit    — Quit
    /help    — Show commands
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from agent.orchestrator import AgentSession

console = Console()


def _print_response(response) -> None:
    """Pretty-print the agent response with sources and handoff indicator."""
    console.print()
    console.print(Panel(Markdown(response.text), title="[bold cyan]Aster & Row Support[/bold cyan]", border_style="cyan"))

    if response.sources:
        src_text = Text("  Sources: ", style="dim")
        src_text.append(", ".join(response.sources), style="dim italic")
        console.print(src_text)

    if response.handoff:
        console.print(
            "  [bold yellow]⚠  Human handoff recommended[/bold yellow]"
        )
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--session-id", default=None, help="Session ID for logs")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG_LOGGING"] = "true"
        console.print("[dim]Debug logging enabled.[/dim]")

    session = AgentSession(session_id=args.session_id)

    console.print(
        Panel(
            "[bold]Welcome to Aster & Row customer support.[/bold]\n"
            "Ask about returns, shipping, orders, warranty, and more.\n\n"
            "[dim]Commands: /reset  /exit  /help[/dim]",
            title="[bold green]Aster & Row[/bold green]",
            border_style="green",
        )
    )

    while True:
        try:
            raw = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("Goodbye!")
            break

        if raw.lower() == "/reset":
            session.reset()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if raw.lower() == "/help":
            console.print(
                "[dim]/reset — clear conversation history\n"
                "/exit  — quit\n"
                "/help  — show this message[/dim]"
            )
            continue

        response = session.chat(raw)
        _print_response(response)


if __name__ == "__main__":
    main()
