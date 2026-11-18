"""Resumable bounded HTTP ranges; signed URLs can expire during large downloads."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import os
from pathlib import Path
import threading
import time

import requests

from openpi.trlt.tacdex_source import utc_now
from openpi.trlt.tacdex_source import write_json

CHUNK_BYTES = 10 * 1024**2


def download_ranges(destination, url_provider, *, total_bytes, expected_sha256, workers=16):
    """Commit chunks only after exact 206 response size/range and complete pwrite.

    The caller verifies full SHA256 before treating the promoted file as data.
    url_provider is called afresh on HTTP 401/403 and every 30 minutes.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".ranges.part")
    journal = destination.with_suffix(destination.suffix + ".ranges.json")
    count = (total_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
    identity = {"total_bytes": total_bytes, "sha256": expected_sha256, "chunk_bytes": CHUNK_BYTES}
    if journal.exists():
        state = json.loads(journal.read_text())
        if state["identity"] != identity or len(state["done"]) != count:
            raise ValueError("Download range provenance differs")
        if not partial.exists():
            raise ValueError("Range journal exists without partial file")
    else:
        if partial.exists():
            raise ValueError("Partial file exists without a range journal")
        partial.touch(exist_ok=False)
        state = {"identity": identity, "done": [False] * count, "created_at": utc_now()}
        write_json(journal, state)
    url_lock, io_lock = threading.Lock(), threading.Lock()
    url_state = {"url": None, "created": 0.0}
    stopped = threading.Event()

    def get_url(*, force=False):
        with url_lock:
            if force or url_state["url"] is None or time.monotonic() - url_state["created"] > 1800:
                url_state.update(url=url_provider(), created=time.monotonic())
            return url_state["url"]

    descriptor = os.open(partial, os.O_RDWR)

    def fetch(index):
        start, end = index * CHUNK_BYTES, min((index + 1) * CHUNK_BYTES, total_bytes)
        error = None
        for attempt in range(6):
            if stopped.is_set():
                raise RuntimeError("Download stopped after another chunk failed")
            try:
                with requests.get(
                    get_url(), headers={"Range": f"bytes={start}-{end - 1}"}, stream=True, timeout=(30, 180)
                ) as response:
                    if response.status_code in (401, 403):
                        get_url(force=True)
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end - 1}/{total_bytes}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server did not honor the exact requested byte range")
                    data = response.content
                if len(data) != end - start:
                    raise ValueError("Truncated HTTP range")
                offset = 0
                while offset < len(data):
                    written = os.pwrite(descriptor, data[offset:], start + offset)
                    if not written:
                        raise OSError("Incomplete local write")
                    offset += written
                return index
            except (OSError, ValueError, requests.RequestException) as exc:
                error = exc
                if attempt < 5:
                    time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"Range {index} failed after six attempts") from error

    missing = [i for i, done in enumerate(state["done"]) if not done]
    print(
        f"Range download: reusing {count - len(missing)}/{count} committed chunks; "
        f"fetching {len(missing)} chunks with {workers} workers",
        flush=True,
    )
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(fetch, index) for index in missing]
        last_print = 0.0
        for future in as_completed(futures):
            index = future.result()
            # Flush data before committing its journal bit. A process/power
            # interruption can only cause an uncommitted chunk to be retried.
            with io_lock:
                os.fsync(descriptor)
                state["done"][index] = True
                state["updated_at"] = utc_now()
                write_json(journal, state)
            if time.monotonic() - last_print > 30:
                completed = sum(
                    min(CHUNK_BYTES, total_bytes - i * CHUNK_BYTES) for i, done in enumerate(state["done"]) if done
                )
                print(f"Downloaded/retained {completed / 1e9:.2f}/{total_bytes / 1e9:.2f} GB", flush=True)
                last_print = time.monotonic()
    except BaseException:
        stopped.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        os.close(descriptor)
    if not all(state["done"]) or partial.stat().st_size != total_bytes:
        raise ValueError("Range download is incomplete")
    partial.replace(destination)
    state.update(status="downloaded; full SHA256 verification required", completed_at=utc_now())
    write_json(journal, state)
