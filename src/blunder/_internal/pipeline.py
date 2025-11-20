"""Scalable PGN processing pipeline skeleton.

This module sketches out a multiprocessing-based pipeline that reads games
from a PGN file using a pre-computed index, performs engine evaluations, and
persists structured results into a SQL database. Replace the TODO placeholders
with concrete implementations that suit your environment.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from chess import engine, pgn

from blunder._internal.util import discover_offsets, load_offsets

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency guard
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class EngineConfig:
    """Configuration for the chess engine backend."""

    executable_path: Path
    config_hash_mb: int = 512
    config_threads: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
    config_multipv: int = 1
    config_ponder: bool = False
    config_show_wdl: bool = True
    depth: int = 14
    info: engine.Info = field(default_factory=lambda: engine.INFO_SCORE)
    # TODO: Add any other engine-specific options you need


@dataclass(slots=True)
class ProcessingConfig:
    """High-level knobs for the parallel processing pipeline."""

    pgn_path: Path
    index_path: Path | None = None
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1) - 1))
    batch_size: int = 64
    max_games: int | None = None
    show_progress: bool = True
    chunk_size: int = 8192
    engine: EngineConfig | None = None
    # database_url: str = "sqlite:///analysis.db"
    output_parquet_dir: Path = Path("analysis_data")  # NEW: Replace database_url
    parquet_partition_cols: list[str] = field(default_factory=lambda: ["time_control"])  # NEW
    parquet_compression: str = "zstd"  # NEW: Better than snappy for text data
    parquet_row_group_size: int = 100_000  # NEW: Tune for query patterns
    checkpoint_path: Path | None = None
    resume_from_checkpoint: bool = True
    # TODO: Add tuning parameters (e.g., time management, pruning strategy)

def time_control_type(initial_time: int, increment: int) -> pgn.TimeControlType:
    """Determine time control category by estimated total time (seconds).

    Uses named constants for the thresholds to avoid magic numbers.
    """
    # Thresholds (estimated total time in seconds) for categories
    bullet_max = 179
    blitz_max = 479
    rapid_max = 1499

    estimated_time = initial_time + (40 * increment)
    if estimated_time <= bullet_max:
        return pgn.TimeControlType.BULLET
    if estimated_time <= blitz_max:
        return pgn.TimeControlType.BLITZ
    if estimated_time <= rapid_max:
        return pgn.TimeControlType.RAPID
    return pgn.TimeControlType.STANDARD


def parse_time_control(tc_str: str) -> tuple[pgn.TimeControlType, int, int]:
    """Parse a time control string into initial time and increment in seconds."""
    tc = (tc_str or "").strip()
    # Common sentinel values indicating no classical time control
    if not tc or tc.lower() in ("unlimited", "-", "?", "*"):
        return (pgn.TimeControlType.UNLIMITED, 0, 0)

    # Standard "initial+increment" form
    if "+" in tc:
        try:
            initial_str, increment_str = tc.split("+", 1)
            initial = int(initial_str)
            increment = int(increment_str)
            return (time_control_type(initial, increment), initial, increment)
        except Exception:  # noqa: BLE001
            logger.warning("Malformed time control '%s'; falling back to UNLIMITED", tc)
            return (pgn.TimeControlType.UNLIMITED, 0, 0)

    # Single number like "300" -> treat as initial seconds with 0 increment
    if tc.isdigit():
        initial = int(tc)
        return (time_control_type(initial, 0), initial, 0)

    # Try best-effort extraction (e.g., weird annotations)
    import re  # noqa: PLC0415

    m = re.search(r"(\d+)(?:\D+(\d+))?", tc)
    if m:
        initial = int(m.group(1))
        increment = int(m.group(2)) if m.group(2) else 0
        return (time_control_type(initial, increment), initial, increment)

    logger.warning("Unrecognized time control '%s'; treating as UNLIMITED", tc)
    return (pgn.TimeControlType.UNLIMITED, 0, 0)


class ParquetWriter:
    """Manages writing game records to worker-specific Parquet files."""

    def __init__(
        self,
        output_dir: Path,
        compression: str = "zstd",
        row_group_size: int = 100_000,
    ) -> None:
        self.output_dir = output_dir
        self.compression = compression
        self.row_group_size = row_group_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_batch(self, records: Sequence[dict[str, Any]], worker_id: int) -> tuple[Path, int]:
        """Write batch to worker-specific Parquet file.

        Returns:
            Tuple of (output_path, max_offset_in_batch)
        """
        if not records:
            raise ValueError("Cannot write empty batch")

        table = pa.Table.from_pylist(records)

        # Worker-specific filename to avoid conflicts
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        first_offset = records[0]["offset"]
        filename = f"batch_w{worker_id:02d}_{timestamp}_{first_offset}.parquet"
        output_path = self.output_dir / filename

        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            use_dictionary=True,  # Excellent for repeated strings (ECO, FEN prefixes)
            write_statistics=True,  # Enable predicate pushdown
            row_group_size=self.row_group_size,
        )

        max_offset = max(record["offset"] for record in records)
        game_total = len({record["offset"] for record in records})
        logger.debug(
            f"Worker {worker_id}: Wrote {len(records)} moves from {game_total} games to {output_path}",
        )
        return output_path, max_offset

@dataclass(slots=True)
class OffsetCheckpoint:
    """Track the highest processed PGN offset to support resumable runs."""

    path: Path
    value: int | None = None

    @classmethod
    def load(cls, path: Path) -> OffsetCheckpoint:
        """Create a checkpoint tracker from disk, ignoring corrupt payloads."""
        value: int | None = None
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                payload = json.loads(raw)
                stored = payload.get("max_offset")
                if isinstance(stored, int):
                    value = stored
            except (OSError, json.JSONDecodeError, ValueError):  # pragma: no cover - defensive guard
                logger.warning("Failed to read checkpoint at %s; starting fresh", path, exc_info=True)
        return cls(path=path, value=value)

    def filter_offsets(self, offsets: np.ndarray) -> np.ndarray:
        """Drop offsets that were already processed according to the checkpoint."""
        if self.value is None:
            return offsets
        mask = offsets > self.value
        skipped = int(offsets.size - mask.sum())
        if skipped:
            logger.info(f"Skipping {skipped} games at or below checkpoint offset {self.value}")
        return offsets[mask]

    def update_from_max_offset(self, max_offset: int) -> None:  # NEW: Simpler signature
        """Advance checkpoint when a batch is successfully written."""
        if self.value is None or max_offset > self.value:
            self.value = max_offset
            self._write()

    def _write(self) -> None:
        payload = json.dumps({"max_offset": self.value})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.path.parent),
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = Path(tmp.name)
        os.replace(temp_path, self.path)


def ensure_index(config: ProcessingConfig) -> np.ndarray:
    """Ensure a valid offset index is available and return it as a numpy array."""
    if config.index_path and config.index_path.exists():
        logger.info("Loading offsets from %s", config.index_path)
        offsets = load_offsets(str(config.index_path), as_numpy=True, use_memmap=True)
    else:
        logger.info(f"Index file not provided. Discovering offsets for {config.pgn_path}")
        offsets = discover_offsets(
            str(config.pgn_path),
            show_progress=config.show_progress,
            as_numpy=True,
            use_memmap=True,
            memmap_path=str(config.index_path or config.pgn_path.with_suffix(".idx")),
        )
    if not isinstance(offsets, np.ndarray):
        raise TypeError("Expected offsets to be a numpy.ndarray")
    return offsets


def game_offsets(offsets: np.ndarray, *, max_games: int | None = None) -> np.ndarray:
    """Slice offsets if ``max_games`` is specified."""
    if max_games is not None:
        return offsets[:max_games]
    return offsets


def chunk_offsets(offsets: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Generate contiguous chunks of offsets to balance worker load."""
    total = offsets.size
    for start in range(0, total, batch_size):
        yield offsets[start : start + batch_size]


def parse_game(
    game: pgn.Game,
    offset: int,
    chess_engine: engine.SimpleEngine,
    engine_limit: engine.Limit,
    engine_info: engine.Info,
) -> list[dict[str, Any]]:
    """Parse a PGN game into per-move dictionaries with duplicated game metadata."""
    board = game.board()

    site = game.headers.get("Site", "")
    game_id = site.split("/")[-1]
    white_elo = int(game.headers.get("WhiteElo", "0"))
    black_elo = int(game.headers.get("BlackElo", "0"))
    white_rating_diff = int(game.headers.get("WhiteRatingDiff", "0"))
    black_rating_diff = int(game.headers.get("BlackRatingDiff", "0"))
    eco = game.headers.get("ECO", "")
    time_control_type, game_time, increment = parse_time_control(game.headers.get("TimeControl", ""))
    termination = game.headers.get("Termination", "")

    result_str = game.headers.get("Result", "")
    result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
    result = result_map.get(result_str, 0)

    rows: list[dict[str, Any]] = []

    info_before = chess_engine.analyse(board, engine_limit, info=engine_info)  # type: ignore  # noqa: PGH003
    score_before = info_before.get("score")

    if not score_before:
        raise ValueError("Engine did not return a score for the initial position")

    score_before_w = score_before.white().score(mate_score=100000)
    logger.debug(f"Initial position score: {score_before_w}")

    for node in game.mainline():
        move = node.move
        move_num = board.fullmove_number
        move_san = board.san(move)
        turn = board.turn
        logger.debug(f"Analyzing move {move_num}.{'w' if turn else 'b'}: {move_san}")

        wdl_obj = info_before.get("wdl")
        wdl_before = wdl_obj.white() if wdl_obj else engine.Wdl(0, 1000, 0)
        logger.debug(
            f"WDL before move: W={wdl_before.winning_chance():.2%}, D={wdl_before.drawing_chance():.2%}, L={wdl_before.losing_chance():.2%}",
        )

        board.push(move)

        # Position after move
        info_after = chess_engine.analyse(board, engine_limit, info=engine_info)  # type: ignore  # noqa: PGH003
        score_after = info_after.get("score")
        if not score_after:
            raise ValueError("Engine did not return a score after move")
        score_after_w = score_after.white().score(mate_score=100000)
        mate = score_after.white().mate()
        mate_in = float(mate) if mate else float("inf")
        clock = node.clock()
        piece_moved = board.piece_at(move.to_square)
        board_fen = board.board_fen()
        is_check = board.is_check()
        eval_delta = score_after_w - score_before_w

        rows.append(
            {
                "game_id": game_id,
                "offset": offset,
                "site": site,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "white_rating_diff": white_rating_diff,
                "black_rating_diff": black_rating_diff,
                "eco": eco,
                "time_control": time_control_type.value,
                "game_time": game_time,
                "increment": increment,
                "result": result,
                "termination": termination,
                "turn": turn,
                "move": move_san,
                "fullmove_number": move_num,
                "cp_score": score_after_w,
                "winning_chance": float(wdl_before.winning_chance()),
                "drawing_chance": float(wdl_before.drawing_chance()),
                "losing_chance": float(wdl_before.losing_chance()),
                "piece_moved": piece_moved.piece_type if piece_moved else None,
                "board_fen": board_fen,
                "is_check": is_check,
                "mate_in": mate_in,
                "clock": float(clock) if clock is not None else None,
                "eval_delta": eval_delta,
            },
        )

        info_before = info_after
        score_before_w = score_after_w

    return rows


def worker_initialize(engine_config: EngineConfig | None) -> engine.SimpleEngine:
    """Optional initializer executed once per worker process."""
    if engine_config is None:
        raise NotImplementedError("Engine configuration is required")

    logger.debug("Worker initializing engine resources: %s", engine_config)
    stockfish = engine.SimpleEngine.popen_uci(str(engine_config.executable_path))
    stockfish.configure(
        {
            "Threads": engine_config.config_threads,
            "Hash": engine_config.config_hash_mb,
            "UCI_ShowWDL": engine_config.config_show_wdl,
        },
    )

    return stockfish


def worker_process(
    pgn_path: Path,
    offsets: Sequence[int],
    engine_config: EngineConfig | None,
    parquet_writer: ParquetWriter,  # NEW: Pass writer instance
    worker_id: int,  # NEW: For unique filenames
) -> tuple[int, Path, int]:  # NEW: Return (game_count, output_path, max_offset)
    """Process batch and write directly to Parquet from worker.

    Returns:
        Tuple of (game_count, output_path, max_offset_in_batch)
    """
    if engine_config is None:
        raise NotImplementedError("Engine configuration is required")

    try:
        min_offset = offsets[0]
        max_offset = offsets[-1]
        logger.debug(f"Worker processing offsets: first={min_offset} last={max_offset} count={len(offsets)}")

        records: list[dict[str, Any]] = []
        game_count = 0

        stockfish = worker_initialize(engine_config)

        with open(pgn_path, encoding="utf-8") as handle:
            handle.seek(min_offset)
            while handle.tell() <= max_offset:
                curr_offset = handle.tell()
                curr_game = pgn.read_game(handle)
                if curr_game is None:
                    logger.debug(f"Reached EOF while reading games up to offset {max_offset}")
                    break
                if curr_game.errors:
                    logger.warning(f"Encountered errors in game at offset {curr_offset}: {curr_game.errors}")
                    continue
                engine_limit = engine.Limit(depth=engine_config.depth)
                engine_info = engine_config.info
                parsed_moves = parse_game(curr_game, curr_offset,stockfish, engine_limit, engine_info)
                if not parsed_moves:
                    continue
                records.extend(parsed_moves)
                game_count += 1

            output_path, batch_max_offset = parquet_writer.write_batch(records, worker_id)
            stockfish.quit()
            return game_count, output_path, batch_max_offset

    except Exception:
        logger.exception("Worker failed while processing offsets: first=%s", offsets[0])
        raise


def process_batches(
    config: ProcessingConfig,
    offsets: np.ndarray,
) -> Iterator[tuple[int, Path, int]]:  # NEW: Return (game_count, path, max_offset)
    """Dispatch batches with workers writing their own Parquet files."""
    engine_config = config.engine
    parquet_writer = ParquetWriter(
        config.output_parquet_dir,
        config.parquet_compression,
        config.parquet_row_group_size,
    )

    worker_fn = partial(
        worker_process,
        config.pgn_path,
        engine_config=engine_config,
        parquet_writer=parquet_writer,
    )

    with ProcessPoolExecutor(
        max_workers=config.workers,
    ) as pool:
        futures = []
        for worker_id, batch_offsets in enumerate(chunk_offsets(offsets, config.batch_size)):
            futures.append(pool.submit(worker_fn, batch_offsets.tolist(), worker_id=worker_id))

        for future in as_completed(futures):
            yield future.result()


def report_progress(
    iterable: Iterable[tuple[int, Path]],  # NEW: Accept (game_count, path)
    total: int,
    *,
    show_progress: bool,
) -> Iterator[tuple[int, Path]]:
    """Wrap iterable with progress reporter when enabled."""
    if not show_progress:
        yield from iterable
        return

    if tqdm is None:
        logger.warning("tqdm unavailable; no progress bar")
        yield from iterable
        return

    with tqdm(total=total, unit="game", desc="Processing") as progress:
        for game_count, path in iterable:
            progress.update(game_count)
            yield game_count, path


def track_results(
    results_iter: Iterable[tuple[int, Path, int]],  # NEW: (game_count, path, max_offset)
    checkpoint: OffsetCheckpoint | None = None,
) -> Iterator[tuple[int, Path]]:  # Yield for progress tracking
    """Track Parquet writes and update checkpoint."""
    for game_count, output_path, max_offset in results_iter:
        logger.debug(f"Batch written: {game_count} games → {output_path}")
        if checkpoint is not None:
            checkpoint.update_from_max_offset(max_offset)
        yield game_count, output_path


def run_pipeline(config: ProcessingConfig) -> None:
    """Main entry point coordinating the full pipeline."""
    logger.info("Starting PGN processing pipeline for %s", config.pgn_path)
    offsets = ensure_index(config)
    offsets = game_offsets(offsets, max_games=config.max_games)

    checkpoint: OffsetCheckpoint | None = None
    if config.resume_from_checkpoint:
        checkpoint_path = config.checkpoint_path or config.pgn_path.with_suffix(".checkpoint.json")
        checkpoint = OffsetCheckpoint.load(checkpoint_path)
        offsets = checkpoint.filter_offsets(offsets)

    total_games = offsets.size
    logger.info("Processing %d games", total_games)

    if total_games == 0:
        logger.info("No new games to process; exiting")
        return

    # REMOVE: db_manager = DatabaseSessionManager(...)
    # REMOVE: db_manager.initialize()

    result_batches = process_batches(config, offsets)  # Returns (count, path, max_offset)
    result_batches = track_results(result_batches, checkpoint)  # Update checkpoint
    result_batches = report_progress(result_batches, total_games, show_progress=config.show_progress)

    # Consume iterator to execute pipeline
    output_files = list(result_batches)

    logger.info("Pipeline completed successfully")
    logger.info(f"Wrote {len(output_files)} Parquet files to {config.output_parquet_dir}")


def consolidate_parquet_files(
    input_dir: Path,
    output_dir: Path,
    partition_cols: list[str] | None = None,
) -> None:
    """Merge worker-specific files into partitioned dataset (optional post-processing)."""
    try:
        import polars as pl  # noqa: PLC0415
    except ImportError:
        logger.warning("polars not installed; skipping consolidation")
        return

    logger.info(f"Consolidating Parquet files from {input_dir}")
    df = pl.scan_parquet(f"{input_dir}/*.parquet")

    if partition_cols:
        df.sink_parquet(
            pl.PartitionByKey(output_dir, by=partition_cols),
            compression="zstd",
            row_group_size=100_000,
        )
    else:
        df.sink_parquet(output_dir, compression="zstd")

    logger.info(f"Consolidated dataset written to {output_dir}")


def main(pgn_path: Path, idx_path: Path | None, engine_path: Path) -> None:
    """Example CLI harness for running the pipeline."""
    # TODO: Replace with argparse-based CLI that constructs configs from user input
    raise NotImplementedError("CLI entry point is not implemented")


if __name__ == "__main__":  # pragma: no cover - import-side execution guard
    # raise SystemExit("The pipeline module exposes no standalone CLI entry point.")
    pgn_path = Path("/home/vandy/Work/MATH6310/blunder-analysis/data/raw/lichess_db_standard_rated_2025-08.pgn")
    index_path = Path("/home/vandy/Work/MATH6310/blunder-analysis/data/raw/lichess_db_standard_rated_2025-08.idx")

    engine_path = os.environ.get("STOCKFISH_PATH", "")
    engine_config = EngineConfig(executable_path=Path(engine_path))
    engine_config.config_hash_mb = 512
    engine_config.config_threads = 1
    engine_config.depth = 14

    config = ProcessingConfig(
        pgn_path=pgn_path,
        index_path=index_path,
        workers=14,
        batch_size=128,
        max_games=55_000,
        resume_from_checkpoint=True,
        engine=engine_config,
        output_parquet_dir=Path("/home/vandy/Work/MATH6310/blunder-analysis/data/silver"),
        checkpoint_path=Path("/home/vandy/Work/MATH6310/blunder-analysis/data/checkpoint/checkpoint.json"),
    )

    run_pipeline(config)
