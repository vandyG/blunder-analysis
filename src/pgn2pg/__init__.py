"""pgn2pg package.

PGN to Postgress conversion
"""

from __future__ import annotations

from pgn2pg._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
