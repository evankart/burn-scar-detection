"""Bytes this app has actually received from the network.

The OS-wide network counter includes every other program's traffic, so it overstates
what the app downloads. This module counts only this process:

* macOS: poll `nettop` for our own PID and add up growth in bytes received per
  non-loopback connection (loopback is just the browser talking to Streamlit).
* elsewhere (Linux, including the Hugging Face Space container): the interface counter,
  which inside a container only sees the container's own traffic.

Use start() once, then total() for a monotonically increasing byte count.
"""
import os
import platform
import subprocess
import threading

_lock = threading.Lock()
_total = 0
_started = False
_psutil_base = None


def _loopback(conn: str) -> bool:
    remote = conn.split("<->")[-1]
    return remote.startswith(("127.", "::1", "localhost", "fe80"))


_last: dict = {}        # connection -> last bytes_in seen
_last_poll = 0.0


def _poll(count: bool = True) -> None:
    """Read nettop once for our PID and add each connection's growth since last time.
    Tracking per connection keeps the total monotonic when connections close."""
    global _total, _last_poll
    import time
    try:
        out = subprocess.run(
            ["nettop", "-p", str(os.getpid()), "-x", "-L", "1", "-J", "bytes_in"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return
    _last_poll = time.time()
    with _lock:
        for line in out.splitlines():
            if "<->" not in line:
                continue
            parts = line.split(",")
            try:
                name, n = parts[0], int(parts[1])
            except (IndexError, ValueError):
                continue
            if _loopback(name):
                continue
            prev = _last.get(name, 0)
            if n > prev and count:
                _total += n - prev
            _last[name] = max(n, prev)


def start() -> None:
    global _started, _psutil_base
    if _started:
        return
    _started = True
    if platform.system() == "Darwin":
        _poll(count=False)      # baseline: ignore bytes from before we started
        return
    try:
        import psutil
        _psutil_base = psutil.net_io_counters().bytes_recv
    except Exception:
        _psutil_base = None


def total() -> int:
    """Bytes received since start()."""
    import time
    if platform.system() == "Darwin":
        if time.time() - _last_poll > 0.25:
            _poll()
        with _lock:
            return _total
    if _psutil_base is not None:
        import psutil
        return psutil.net_io_counters().bytes_recv - _psutil_base
    return 0
