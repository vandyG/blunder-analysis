# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m pgn2pg` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `pgn2pg.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `pgn2pg.__main__` in `sys.modules`.

from __future__ import annotations
from typing_extensions import Annotated

import argparse
import sys
from typing import Any, Optional

from pgn2pg._internal import debug

import typer

app = typer.Typer()


class _DebugInfo(argparse.Action):
    def __init__(self, nargs: int | str | None = 0, **kwargs: Any) -> None:
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        debug._print_debug_info()
        sys.exit(0)


def get_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    parser = argparse.ArgumentParser(prog="pgn2pg")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {debug._get_version()}")
    parser.add_argument("--debug-info", action=_DebugInfo, help="Print debug information.")
    return parser

@app.command()
def main(version: Annotated[bool | None, typer.Option("--version", callback=debug._get_version)]) -> int:
    """Run the main program.

    This function is executed when you type `pgn2pg` or `python -m pgn2pg`.

    Parameters:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    print("Welcome to pgn2pg.")
