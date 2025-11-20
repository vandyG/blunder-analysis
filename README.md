# blunder-analysis

[![ci](https://github.com/vandyG/blunder-analysis/workflows/ci/badge.svg)](https://github.com/vandyG/blunder-analysis/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://vandyG.github.io/blunder-analysis/)
[![pypi version](https://img.shields.io/pypi/v/blunder.svg)](https://pypi.org/project/blunder/)
[![gitter](https://img.shields.io/badge/matrix-chat-4DB798.svg?style=flat)](https://app.gitter.im/#/room/#blunder-analysis:gitter.im)

Blunder analysis in Chess

## Installation

```bash
pip install blunder
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install blunder
```

## Usage

After installation you can run the CLI directly:

```bash
# when installed via pip
blunder --help
```

If you haven't installed the package (for example when running from the repository), invoke the package with Python:

```bash
# run the same CLI from source
python -m blunder --help
```

Note: The project declares a console script entry point (`blunder = "blunder:app"`) in `pyproject.toml`. This makes the `blunder` command available after installation; when working from the repo without installing, use `python -m blunder` to invoke the same application.

**Commands, Outputs, and Schema**

- **make-offset**: Create a binary index of byte offsets for each game in a PGN file.
	- **Usage:** `blunder make-offset <pgn_path>` or `python -m blunder make-offset <pgn_path>`.
	- **Output:** A binary `.idx` file next to the PGN (e.g. `games.idx`) containing consecutive 64-bit integers (`np.int64`) representing byte offsets. This file is created by `discover_offsets(..., use_memmap=True)` and may be memory-mapped by subsequent runs.

- **evaluate**: Run the processing pipeline that evaluates positions with an engine and writes per-move records to Parquet files.
	- **Usage:** `blunder evaluate <pgn_path> [OPTIONS]` or `python -m blunder evaluate <pgn_path> [OPTIONS]`.
	- **Required environment:** Set `STOCKFISH_PATH` to the Stockfish binary path (e.g. `export STOCKFISH_PATH=/usr/bin/stockfish`). The CLI will raise an error if this is not set.
	- **Important options:** `--workers/-W` (process count), `--batch-size/-B`, `--max-games/-M`, `--index-path`, `--output-path`, `--checkpoint-path`, engine tuning options like `--depth/-D`, `--threads/-T`, `--hash-mb/-H`, `--multipv/-P`, `--info` (engine Info), `--resume`, and `--quiet`.
	- **Behavior:** The pipeline ensures an index is present (loads `--index-path` or creates `<pgn>.idx`), optionally loads a checkpoint JSON (`{"max_offset": <int>}`) when `--resume` is enabled, spawns worker processes which initialize a Stockfish engine, parse games, call the engine for evaluations, and write worker-specific Parquet batches.

- **main**: A Typer command stub exists but `main` currently raises `NotImplementedError` (not usable yet).

**Generated files & formats**

- `*.idx` (index): Binary file of 64-bit integers (NumPy `int64`) holding byte offsets for each game's header in the PGN. Loadable via `blunder` internals or `numpy.fromfile` / `np.memmap`.
- `*.checkpoint.json`: JSON checkpoint with shape `{"max_offset": <int>}` (UTF-8 text). Default path is the PGN path with suffix `.checkpoint.json` unless `--checkpoint-path` is provided. The pipeline atomically updates this file after batches complete.
- `batch_wXX_YYYYmmdd_HHMMSS_FIRSTOFFSET.parquet`: Worker-specific Parquet outputs. Filenames use the worker id, UTC timestamp, and the first offset in the batch (e.g. `batch_w01_20251116_175201_0.parquet`). Default output directory is the PGN parent directory unless `--output-path` is provided.

Writer settings (defaults, configurable via `ProcessingConfig`): compression `zstd`, `use_dictionary=True`, `write_statistics=True`, `row_group_size=100000`.

Optional consolidation: use `consolidate_parquet_files(input_dir, output_dir, partition_cols)` to merge worker files into a partitioned dataset. This uses `polars` (if installed) and respects `parquet_partition_cols` (default `["time_control"]`).

**Parquet schema (per-move records)**

Each row in the Parquet files is a per-move dictionary produced by `parse_game()`; fields and types are:

- `game_id`: string — game identifier (extracted from `Site` header's suffix).
- `offset`: int64 — byte offset in PGN where the game begins.
- `site`: string — `Site` header.
- `white_elo`: int — White player's Elo.
- `black_elo`: int — Black player's Elo.
- `white_rating_diff`: int
- `black_rating_diff`: int
- `eco`: string — ECO code.
- `time_control`: int — numeric value from `pgn.TimeControlType.value`.
- `game_time`: int — initial time (seconds).
- `increment`: int — increment (seconds).
- `result`: int — 1 (White win), -1 (Black win), 0 (draw/unknown).
- `termination`: string
- `turn`: bool — `True` for White to move, `False` for Black.
- `move`: string — SAN of the move.
- `fullmove_number`: int
- `cp_score`: int — centipawn score (white-perspective) for the position after the move.
- `winning_chance`: float — engine WDL winning chance before the move.
- `drawing_chance`: float
- `losing_chance`: float
- `piece_moved`: int | null — numeric `chess.PieceType` moved or null.
- `board_fen`: string — board FEN (board part only).
- `is_check`: bool
- `mate_in`: float — mate distance (float('inf') when none).
- `clock`: float | null — remaining clock seconds, if present.
- `eval_delta`: int — change in centipawn score (post - pre move).

Note: types above map to typical PyArrow types when writing with `pyarrow.Table.from_pylist()` (strings, `int64`, `float64`, `bool`, nullable fields where needed).

**Quick examples (fish shell)**

Set engine path and create an index:
```fish
set -x STOCKFISH_PATH /usr/bin/stockfish
python -m blunder make-offset data/raw/lichess_db_standard_rated_2025-08.pgn
```

Run evaluation for first 1000 games with 4 workers and write to `data/silver`:
```fish
python -m blunder evaluate data/raw/lichess_db_standard_rated_2025-08.pgn --workers 4 --batch-size 128 --max-games 1000 --depth 14 --output-path data/silver --checkpoint-path data/checkpoint/checkpoint.json
```

Inspect a generated Parquet file with `pyarrow`:
```python
import pyarrow.parquet as pq
tbl = pq.read_table('data/silver/batch_w00_20251116_175201_0.parquet')
print(tbl.schema)
print(tbl.to_pandas().head())
```

**Notes & caveats**

- The `main` CLI command is a placeholder and raises `NotImplementedError`.
- Some internals (e.g., `evaluate_game`, database insertion helpers, and `_read_single_game`) contain `NotImplementedError` placeholders; the current pipeline writes Parquet files via `ParquetWriter` rather than inserting into a SQL database.
- Consolidation into a partitioned dataset requires `polars`.

