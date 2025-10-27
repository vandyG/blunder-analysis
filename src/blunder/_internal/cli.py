# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m blunder` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `blunder.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `blunder.__main__` in `sys.modules`.

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Annotated, Any

import typer

from blunder._internal import debug
from blunder._internal.util import discover_offsets

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Blunder command line interface.")


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
    parser = argparse.ArgumentParser(prog="blunder")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {debug._get_version()}")
    parser.add_argument("--debug-info", action=_DebugInfo, help="Print debug information.")
    return parser


@app.command()
def main(
    version: Annotated[bool | None, typer.Option("--version", callback=debug._get_version, is_eager=True)] = None,
    debug_info: Annotated[
        bool | None,
        typer.Option("--debug-info", callback=debug._print_debug_info, is_eager=True),
    ] = None,
) -> int:
    """Run the main program.

    This function is executed when you type `blunder` or `python -m blunder`.

    Parameters:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    raise NotImplementedError("Main command not implemented yet.")


@app.command()
def pgn2pg(
    pgn_path: Path,
    *,
    create_index_only: Annotated[
        bool,
        typer.Option("--create-index-only", "-i", help="Create only index file."),
    ] = False,
    use_memmap: Annotated[bool, typer.Option("--use-memmap", "-m", help="Use memory-mapped files.")] = True,
) -> None:
    """Convert PGN files to PG format.

    This is a placeholder for the actual implementation.
    """
    print("pgn2pg command invoked.")

    memmap_path = str(pgn_path.with_suffix(".idx")) if use_memmap else None

    index = discover_offsets(str(pgn_path), use_memmap=use_memmap, memmap_path=memmap_path)

    if create_index_only:
        print(f"Index created with {len(index)} entries.")
        raise typer.Exit
