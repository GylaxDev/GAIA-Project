"""Zero-copy process-to-renderer transport for large NumPy mesh buffers.

Sending 50–100 MB through ProcessPoolExecutor's pickle pipe pauses the render
GIL during unpickling. Protocol-5 buffers are instead staged in one temporary
mapped file; only the small object graph travels through the pipe. Array views
retain the mapping. Streaming mappings transfer their final cleanup to a worker
after the last view is released, so Windows unmapping cannot stall a frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import threading


_cleanup_pool = None
_cleanup_lock = threading.Lock()


def _close_mapping(mapping):
    mapping.close()


def _ensure_cleanup_pool():
    global _cleanup_pool
    with _cleanup_lock:
        if _cleanup_pool is None:
            _cleanup_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mapped-cleanup')
            # open() runs in a chunk worker; start the cleanup thread here, not
            # when the render thread releases its first completed upload.
            _cleanup_pool.submit(lambda: None)


class _DeferredMapping(mmap.mmap):
    """All buffer views retain this owner; finalization hands it to the reaper."""
    def __del__(self):
        pool = _cleanup_pool
        if pool is not None and not self.closed:
            try:
                # Holding self in the task postpones native deallocation. This
                # only runs at the last reference, after buffer exports end;
                # never forcibly close a mapping while a NumPy view is alive.
                pool.submit(_close_mapping, self)
            except RuntimeError:
                pass  # executor/interpreter shutdown: native deallocation is safe


def shutdown_staging_cleanup(wait=True):
    """Drain (never cancel) deferred unmaps outside the rendering loop."""
    global _cleanup_pool
    with _cleanup_lock:
        pool, _cleanup_pool = _cleanup_pool, None
    if pool is not None: pool.shutdown(wait=wait)


@dataclass
class StagedResult:
    path: str
    metadata: bytes
    buffers: list

    def open(self, defer_cleanup=False):
        if defer_cleanup: _ensure_cleanup_pool()
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
        if os.name == 'nt': flags |= os.O_TEMPORARY
        fd = os.open(self.path, flags)
        try:
            mapping_type = _DeferredMapping if defer_cleanup else mmap.mmap
            mapping = mapping_type(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        if os.name != 'nt': os.unlink(self.path)
        view = memoryview(mapping)
        result = pickle.loads(self.metadata, buffers=[view[o:o+n] for o, n in self.buffers])
        # Traffic survives the upload. Its small tables must not pin the entire
        # geometry mapping (tens of MB) for the lifetime of a resident tile.
        roads = getattr(result, 'roads', None)
        if roads is not None:
            import numpy as np
            for name, data in vars(roads).items():
                if isinstance(data, np.ndarray) and not data.flags.writeable:
                    setattr(roads, name, data.copy())
        for path in getattr(result, 'paths', None) or ():
            for name in ('points', 'arc', 'z'):
                data = getattr(path, name, None)
                if data is not None and not data.flags.writeable:
                    setattr(path, name, data.copy())
        driving = getattr(result, 'driving', None)
        if driving is not None:
            driving.detach_buffers()
        # Every array the resident minimap keeps must be copied out of the
        # mapping: one leftover view would pin the tile's whole staged file
        # (tens of MB) for as long as the tile stays resident.
        minimap = getattr(result, 'minimap', None)
        if minimap is not None:
            import numpy as np
            for name in ('tris', 'centres', 'kind', 'tint'):
                data = getattr(minimap, name, None)
                if isinstance(data, np.ndarray) and not data.flags.writeable:
                    setattr(minimap, name, data.copy())
        return result

    def discard(self):
        try: os.unlink(self.path)
        except FileNotFoundError: pass


def stage_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix='upload-', suffix='.bin', dir=directory)
    buffers = []
    try:
        with os.fdopen(fd, 'wb') as f:
            def buffer(data):
                raw = data.raw()
                if len(raw) < 65536: return True
                offset = f.tell()
                padding = (-offset) % 64
                if padding: f.write(bytes(padding))
                buffers.append((f.tell(), len(raw)))
                f.write(raw)
                return None
            metadata = pickle.dumps(result, protocol=5, buffer_callback=buffer)
        if not buffers:
            os.unlink(path)
            return result
        return StagedResult(path, metadata, buffers)
    except BaseException:
        try: os.unlink(path)
        except OSError: pass
        raise


def discard_future(future):
    """Clean up a completed result after its requester travelled away."""
    if future.cancelled(): return
    try: result = future.result()
    except Exception: return
    if isinstance(result, StagedResult): result.discard()
