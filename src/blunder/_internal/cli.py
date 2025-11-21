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
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import click
import typer
from chess.engine import Info

from blunder._internal import debug
from blunder._internal.util import discover_offsets

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
def make_offset(
    pgn_path: Path,
) -> None:
    """Create a memory-mapped index file for the given PGN file."""
    memmap_path = str(pgn_path.with_suffix(".idx"))

    discover_offsets(str(pgn_path), use_memmap=True, memmap_path=memmap_path)


def _default_workers() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 2
    return max(1, cpu_count - 2)


@app.command()
def evaluate(
    pgn_path: Path,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-W",
            help="Number of worker processes.",
            rich_help_panel="Processing Configuration",
            default_factory=_default_workers,
        ),
    ],
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-B",
            help="Number of positions to process in a batch.",
            rich_help_panel="Processing Configuration",
        ),
    ] = 128,
    max_games: Annotated[
        int,
        typer.Option(
            "--max-games",
            "-M",
            help="Maximum number of games to process.",
            rich_help_panel="Processing Configuration",
        ),
    ] = 1000,
    index_path: Path | None = None,
    hash_mb: Annotated[
        int,
        typer.Option("--hash-mb", "-H", rich_help_panel="Engine Configuration", help="Hash table size in MB."),
    ] = 512,
    threads: Annotated[
        int,
        typer.Option("--threads", "-T", rich_help_panel="Engine Configuration", help="Number of engine threads."),
    ] = 1,
    multipv: Annotated[
        int,
        typer.Option("--multipv", "-P", rich_help_panel="Engine Configuration", help="Number of principal variations."),
    ] = 1,
    depth: Annotated[
        int,
        typer.Option("--depth", "-D", rich_help_panel="Engine Configuration", help="Search depth."),
    ] = 14,
    info: Annotated[
        str | None,
        typer.Option(
            "--info",
            rich_help_panel="Engine Configuration",
            help="Type of engine info to display.",
            click_type=click.Choice([info.name for info in Info]),
            show_choices=True,
        ),
    ] = Info.SCORE.name,
    *,
    resume: Annotated[
        bool,
        typer.Option(
            help="Resume processing from the last checkpoint.",
            rich_help_panel="Processing Configuration",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help="Suppress progress bar and other output.",
            rich_help_panel="Processing Configuration",
        ),
    ] = False,
    ponder: Annotated[
        bool,
        typer.Option("--ponder", rich_help_panel="Engine Configuration", help="Enable pondering."),
    ] = False,
    show_wdl: Annotated[
        bool,
        typer.Option(
            "--show-wdl/--hide-wdl",
            rich_help_panel="Engine Configuration",
            help="Show WDL statistics in the output.",
        ),
    ] = True,
) -> None:
    """Evaluate positions from a PGN file."""
    from blunder._internal.pipeline import EngineConfig, ProcessingConfig, run_pipeline  # noqa: PLC0415

    engine_path = os.environ.get("STOCKFISH_PATH", "")
    if not engine_path:
        raise ValueError("Please set the STOCKFISH_PATH environment variable to the path of the Stockfish engine.")
    engine_config = EngineConfig(executable_path=Path(engine_path))

    engine_config.config_hash_mb = hash_mb
    engine_config.config_threads = threads
    engine_config.config_multipv = multipv
    engine_config.config_ponder = ponder
    engine_config.depth = depth
    engine_config.config_show_wdl = show_wdl
    if info:
        engine_config.info = Info[info]

    config = ProcessingConfig(
        pgn_path=pgn_path,
        index_path=index_path,
        workers=workers,
        batch_size=batch_size,
        max_games=max_games,
        resume_from_checkpoint=resume,
        show_progress=not quiet,
        engine=engine_config,
        output_parquet_dir=output_path if output_path else pgn_path.parent,
        checkpoint_path=checkpoint_path if checkpoint_path else pgn_path.parent,
    )

    run_pipeline(config)
