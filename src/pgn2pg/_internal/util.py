import contextlib
import os
import tempfile
from typing import Optional

import chess
import numpy as np
from tqdm.auto import tqdm


def discover_offsets(
    path: str,
    encoding: str = "utf-8",
    *,
    show_progress: bool = True,
    as_numpy: bool = True,
    use_memmap: bool = False,
    memmap_path: Optional[str] = None,
) -> list[int] | np.ndarray:
    """Discover byte offsets for each game in a PGN file.

    This function builds the offsets in a compact NumPy int64 array to
    significantly reduce memory usage when indexing very large PGN files.

    Arguments:
        path: Path to the PGN file.
        encoding: File encoding used to open the file.
        show_progress: If True, display a tqdm progress bar.
        as_numpy: If True (default), return a NumPy ndarray of dtype int64.
                  If False, return a Python list[int] for compatibility.

    Returns:
        Either a NumPy ndarray of offsets or a Python list of ints.
    """
    total = os.path.getsize(path)

    # Preallocate a reasonably-sized storage and grow by doubling.
    # Using int64 for compatibility with large files and f.tell() values.
    initial_capacity = 1 << 16  # 65_536 entries to start
    count = 0
    capacity = initial_capacity

    # If using memmap, prepare a backing file (either provided or temporary).
    memmap_filename: Optional[str] = None
    memmap: Optional[np.memmap] = None

    if use_memmap:
        if memmap_path is not None:
            memmap_filename = memmap_path
        else:
            with tempfile.NamedTemporaryFile(prefix="pgn_offsets_", delete=False) as tmp:
                memmap_filename = tmp.name

        if memmap_filename is None:
            raise ValueError("memmap filename could not be determined")

        memmap = np.memmap(memmap_filename, dtype=np.int64, mode="w+", shape=(initial_capacity,))

    else:
        arr = np.empty(initial_capacity, dtype=np.int64)

    with open(path, encoding=encoding, errors="replace") as f:
        pbar = tqdm(total=total, unit="B", unit_scale=True, disable=not show_progress)
        while True:
            start = f.tell()
            headers = chess.pgn.read_headers(f)  # pyright: ignore[reportAttributeAccessIssue] # consumes a whole game
            end = f.tell()
            if headers is None:
                break

            # Ensure capacity
            if count >= capacity:
                new_capacity = capacity * 2

                if use_memmap:
                    # Create a new temporary memmap file, copy contents, then atomically replace
                    with tempfile.NamedTemporaryFile(prefix="pgn_offsets_grow_", delete=False) as tmp:
                        new_name = tmp.name

                    new_mm = np.memmap(new_name, dtype=np.int64, mode="w+", shape=(new_capacity,))
                    # copy existing data
                    new_mm[:capacity] = memmap[:capacity]
                    new_mm.flush()
                    # replace the old file with the new one
                    if memmap is not None:
                        with contextlib.suppress(Exception):
                            memmap.flush()
                    # free the memmap object before replacing the file
                    del memmap
                    if memmap_filename is None:
                        raise ValueError("memmap filename could not be determined")
                    os.replace(new_name, memmap_filename)
                    # reopen memmap to the new, larger file
                    memmap = np.memmap(memmap_filename, dtype=np.int64, mode="r+", shape=(new_capacity,))
                else:
                    new_arr = np.empty(new_capacity, dtype=np.int64)
                    new_arr[:capacity] = arr
                    arr = new_arr

                capacity = new_capacity

            if use_memmap:
                memmap[count] = np.int64(start)
            else:
                arr[count] = np.int64(start)

            count += 1

            pbar.update(end - start)

        pbar.update(total - pbar.n)
        pbar.close()

    if count == 0:
        # No games found
        if use_memmap and memmap is not None:
            # clean up transient memmap file if it was created
            with contextlib.suppress(Exception):
                memmap.flush()
        return np.array([], dtype=np.int64) if as_numpy else []

    # Slice to the actual count. For memmap we keep the backing file; slicing
    # a memmap returns an ndarray view (not a new file). If the caller wants
    # the on-disk file, it's at `memmap_path` (or the returned memmap's base
    # if they inspect it).
    result = memmap[:count] if use_memmap and memmap is not None else arr[:count]

    if as_numpy:
        return result
    return result.tolist()
