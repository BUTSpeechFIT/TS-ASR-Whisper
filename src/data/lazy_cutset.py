import fcntl
import gzip
import itertools
import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import List, Union

import numpy as np
from lhotse.cut import Cut
from lhotse.cut.set import deserialize_cut


class LazyCutReader:
    """Random-access reader over a jsonl(.gz) cut manifest that never holds more than one
    deserialized `Cut` in memory at a time.

    `lhotse.load_manifest` parses the whole manifest into a resident Python object graph
    (one full `Cut`, with all its nested `Supervision`/`Recording` objects, per line). Forking
    that into DataLoader workers is cheap at fork time (copy-on-write), but CPython's reference
    counting dirties a page on every read, so each worker ends up materializing its own private
    copy of an increasing fraction of the corpus over the course of a run -- this is what grew
    persistent-worker RSS to ~13GB/worker on the more_data recipe (10 merged corpora).

    Instead, this builds a small one-time index (byte offset + speaker count per cut, computed
    by briefly deserializing each cut in turn and discarding it -- not retaining it) and reads a
    single cut from disk via seek+readline+deserialize on every `__getitem__`. Only the index
    (ids/offsets/speaker-counts -- flat arrays, nothing nested) is ever resident, so there is
    nothing left for worker processes to duplicate.

    The index and a decompressed copy of the manifest (needed for O(1) `seek`, since gzip
    streams can't be seeked into arbitrarily) are cached next to the source file and reused
    across runs as long as the source manifest hasn't changed.
    """

    def __init__(self, manifest_path: Union[str, Path], min_duration: float = 0.0):
        self.manifest_path = Path(manifest_path)
        self.min_duration = min_duration
        self._data_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lazy_data.jsonl")
        self._index_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lazy_index.json")
        self._lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lazy_index.lock")
        self._offsets: List[int] = []
        self._spk_counts: np.ndarray = np.zeros(0, dtype=np.int64)
        self._n_dropped = 0
        self._post_process = None
        self._load_or_build_index()

    def _open_source(self):
        if self.manifest_path.suffix == ".gz":
            return gzip.open(self.manifest_path, "rt")
        return open(self.manifest_path, "r")

    def _index_is_fresh(self) -> bool:
        if not (self._index_path.exists() and self._data_path.exists()):
            return False
        return self._index_path.stat().st_mtime >= self.manifest_path.stat().st_mtime

    def _try_load_fresh_index(self) -> bool:
        if not self._index_is_fresh():
            return False
        with open(self._index_path) as f:
            meta = json.load(f)
        if meta.get("min_duration") != self.min_duration:
            return False
        self._offsets = meta["offsets"]
        self._spk_counts = np.array(meta["spk_counts"], dtype=np.int64)
        self._n_dropped = meta["n_dropped"]
        return True

    def _load_or_build_index(self):
        if self._try_load_fresh_index():
            return
        # DDP training launches several ranks as independent processes that each construct
        # their own training dataset, so without this lock every rank would race to build (and
        # concurrently overwrite) the same cache files on a cold cache. Whichever rank gets the
        # lock first builds it; the rest block here and then just load what it produced.
        with open(self._lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if self._try_load_fresh_index():
                    return
                self._build_index()
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def _build_index(self):
        offsets = []
        spk_counts = []
        n_dropped = 0
        # Build under temp names and only rename into place once fully written (os.replace is
        # atomic on POSIX within the same directory), so a reader can never observe a partially
        # written cache -- and write/read in binary mode so seek/tell are exact byte offsets
        # (text-mode `tell()` cookies aren't arithmetic byte positions).
        tmp_data_path = self._data_path.with_name(self._data_path.name + f".tmp{os.getpid()}")
        tmp_index_path = self._index_path.with_name(self._index_path.name + f".tmp{os.getpid()}")
        with self._open_source() as src, open(tmp_data_path, "wb") as dst:
            pos = 0
            for line in src:
                raw = json.loads(line)
                # deserialize_cut pops "type" off the dict it's given, so hand it a throwaway
                # copy -- we still need the original `raw` to write the cache line verbatim.
                cut = deserialize_cut(dict(raw))
                if cut.duration < self.min_duration:
                    n_dropped += 1
                    continue
                spk_counts.append(len({s.speaker for s in cut.supervisions}))
                out_line = (line if line.endswith("\n") else line + "\n").encode("utf-8")
                offsets.append(pos)
                dst.write(out_line)
                pos += len(out_line)

        with open(tmp_index_path, "w") as f:
            json.dump({
                "min_duration": self.min_duration,
                "offsets": offsets,
                "spk_counts": spk_counts,
                "n_dropped": n_dropped,
            }, f)

        os.replace(tmp_data_path, self._data_path)
        os.replace(tmp_index_path, self._index_path)

        self._offsets = offsets
        self._spk_counts = np.array(spk_counts, dtype=np.int64)
        self._n_dropped = n_dropped

    @property
    def n_dropped(self) -> int:
        return self._n_dropped

    @property
    def spk_counts(self) -> np.ndarray:
        return self._spk_counts

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Cut:
        with open(self._data_path, "rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
        cut = deserialize_cut(json.loads(line))
        if self._post_process is not None:
            cut = self._post_process(cut)
        return cut

    def map(self, fn) -> "LazyCutReader":
        """Mirrors `CutSet.map`: `fn` is applied to each cut as it's deserialized in
        `__getitem__`, rather than eagerly to every cut up front. Mutates and returns self
        (matching how callers use it here: `cutset = cutset.map(fn)`)."""
        self._post_process = fn
        return self

    def __add__(self, other: Union["LazyCutReader", "ConcatCutReader"]) -> "ConcatCutReader":
        return ConcatCutReader([self]) + other


class ConcatCutReader:
    """Concatenates several `LazyCutReader`s (or a mix of readers) behind one positional index,
    mirroring the `reduce(lambda a, b: a + b, cutsets)` concatenation the eager path uses."""

    def __init__(self, readers: List[Union[LazyCutReader, "ConcatCutReader"]]):
        self.readers = list(readers)
        lengths = [len(r) for r in self.readers]
        self._cum = list(itertools.accumulate(lengths))

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, idx: int) -> Cut:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        r_idx = bisect_right(self._cum, idx)
        local_idx = idx - (self._cum[r_idx - 1] if r_idx > 0 else 0)
        return self.readers[r_idx][local_idx]

    def __add__(self, other: Union[LazyCutReader, "ConcatCutReader"]) -> "ConcatCutReader":
        other_readers = other.readers if isinstance(other, ConcatCutReader) else [other]
        return ConcatCutReader(self.readers + other_readers)
