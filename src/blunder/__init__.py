"""blunder-analysis package.

Blunder analysis in Chess
"""

from __future__ import annotations

from blunder._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
