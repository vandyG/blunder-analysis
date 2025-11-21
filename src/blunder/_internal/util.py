import contextlib
import os
import tempfile
from collections.abc import Iterator
from typing import Optional

import numpy as np
from chess import pgn
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

    Scan the PGN file at ``path`` and return the byte offset of the start of
    each game's headers. Offsets are stored as 64-bit integers to remain
    compatible with very large files. The function grows an internal buffer
    exponentially while reading and can optionally use a disk-backed NumPy
    ``memmap`` to avoid holding the full index in RAM.

    Args:
        path (str): Path to the PGN file.
        encoding (str): File encoding used to open the file. Defaults to
            ``'utf-8'``.
        show_progress (bool): If True, display a ``tqdm`` progress bar while
            scanning. Defaults to ``True``.
        as_numpy (bool): If True (default) return a NumPy ``ndarray`` of
            dtype ``int64``. If False, return a Python ``list[int]``.
        use_memmap (bool): If True, store offsets in a disk-backed NumPy
            ``memmap`` instead of keeping them in RAM. Useful for extremely
            large index sizes. Defaults to ``False``.
        memmap_path (Optional[str]): Path to use for the memmap backing file.
            If ``None`` and ``use_memmap`` is True, a temporary file is
            created.

    Returns:
        numpy.ndarray[int64] or list[int]: An array or list of byte offsets
        pointing to the start of each game's header block. When
        ``use_memmap`` is True and ``as_numpy`` is True, the returned object
        is a ``numpy.memmap`` view into the backing file.

    Raises:
        ValueError: If a memmap filename cannot be determined when
            ``use_memmap`` is True.

    Notes:
        Games are read using :mod:`python-chess`'s PGN reader which consumes
        the full game while scanning, so the file handle advances as games
        are discovered. The function guarantees ``dtype=int64`` for
        compatibility with large files and values returned by ``file.tell()``.
        When ``use_memmap`` is used, the backing file path is the provided
        ``memmap_path`` or a temporary file created by this function; callers
        can inspect a returned memmap's ``base`` attribute to locate the
        on-disk file.

    Examples:
        >>> offsets = discover_offsets("games.pgn")
        >>> isinstance(offsets, numpy.ndarray)
        True
        >>> offsets[0]
        0
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
            headers = pgn.read_headers(f)  # pyright: ignore[reportAttributeAccessIssue] # consumes a whole game
            end = f.tell()
            if headers is None:
                break

            # Ensure capacity
            if count >= capacity:
                new_capacity = capacity * 2

                if use_memmap:
                    # Create a new temporary memmap file, copy contents, then atomically replace
                    memmap_dir = None
                    if memmap_filename is not None:
                        memmap_dir = os.path.dirname(os.path.abspath(memmap_filename))
                    with tempfile.NamedTemporaryFile(
                        prefix="pgn_offsets_grow_",
                        dir=memmap_dir,
                        delete=False,
                    ) as tmp:
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

    # If we used a backing memmap file, truncate it to the actual count so
    # later callers using load_offsets(path) see the correct number of entries.
    if use_memmap and memmap is not None and memmap_filename is not None:
        # ensure changes are flushed then truncate file to count * 8 bytes
        with contextlib.suppress(Exception):
            memmap.flush()
            os.truncate(memmap_filename, count * np.dtype(np.int64).itemsize)
        # reopen a tight memmap view (optional)
        result = np.memmap(memmap_filename, dtype=np.int64, mode="r+", shape=(count,))

    if as_numpy:
        return result
    return result.tolist()


def load_offsets(
    path: str,
    *,
    as_numpy: bool = True,
    use_memmap: bool = False,
    chunk_size: Optional[int] = None,
) -> np.ndarray | list[int] | Iterator[int]:
    """Load or stream previously discovered PGN offsets.

    The offsets generated by :func:`discover_offsets` can be stored in a binary
    ``.idx`` file containing consecutive 64-bit integers. This helper provides
    three access patterns:

    * load the entire index into memory as a NumPy array (default)
    * memory-map the file for zero-copy random access (``use_memmap=True``)
    * stream offsets sequentially without retaining them (set ``chunk_size``)

    Args:
        path (str): Path to the binary offsets file.
        as_numpy (bool): When ``True`` (default) return a NumPy array (or
            memmap). When ``False`` return a ``list[int]``. Ignored when
            streaming.
        use_memmap (bool): If ``True`` return a read-only ``numpy.memmap``
            backed by ``path``. Incompatible with ``chunk_size``.
        chunk_size (Optional[int]): When provided, offsets are yielded in a
            streaming fashion ``chunk_size`` elements at a time. Must be
            positive.

    Returns:
        numpy.ndarray | list[int] | Iterator[int]: An array-like collection of
        offsets or an iterator that streams offsets.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file size is not a multiple of ``int64`` bytes or
            mutually exclusive options are provided.

    Examples:
        >>> offsets = load_offsets("games.idx")
        >>> list(load_offsets("games.idx", chunk_size=1024))[:3]
        [0, 128, 256]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer when provided")

    if chunk_size is not None and use_memmap:
        raise ValueError("chunk_size and use_memmap are mutually exclusive")

    dtype_size = np.dtype(np.int64).itemsize
    file_size = os.path.getsize(path)

    if file_size % dtype_size != 0:
        raise ValueError(f"index file {path!r} is not aligned to {dtype_size}-byte boundaries")

    count = file_size // dtype_size

    if chunk_size is not None:

        def _iterator() -> Iterator[int]:
            with open(path, "rb") as handle:
                while True:
                    chunk = np.fromfile(handle, dtype=np.int64, count=chunk_size)
                    if chunk.size == 0:
                        break
                    for value in chunk:
                        yield int(value)

        return _iterator()

    if count == 0:
        empty = np.array([], dtype=np.int64)
        return empty if as_numpy else empty.tolist()

    if use_memmap:
        offsets = np.memmap(path, dtype=np.int64, mode="r", shape=(count,))
        return offsets if as_numpy else offsets.tolist()

    offsets = np.fromfile(path, dtype=np.int64, count=count)
    return offsets if as_numpy else offsets.tolist()
