"""Memory guards that every script calls before it touches the GPU.

A nine day job shares this machine. These limits keep any script in this project from taking the
GPU or the host down. See docs/design.md section 2.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import torch

GPU_FRACTION = 0.25
HOST_BYTES = 3 * 1024**3
MAX_TRAIN_ENVS = 512
MAX_EVAL_ENVS = 256


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _watchdog(host_bytes: int, period: float) -> None:
    while True:
        time.sleep(period)
        rss = _rss_bytes()
        if rss > host_bytes:
            sys.stderr.write(
                f"swarm.compute: host RSS {rss / 2**30:.2f} GB exceeds the cap "
                f"{host_bytes / 2**30:.2f} GB. The process exits now.\n"
            )
            sys.stderr.flush()
            os._exit(3)


def limit_memory(gpu_fraction: float = GPU_FRACTION, host_bytes: int = HOST_BYTES) -> None:
    """Cap GPU memory for this process and start a host memory watchdog.

    The GPU cap makes an allocation past the fraction raise an error instead of taking memory from
    other processes. The watchdog exits the process when resident memory passes the cap. A script
    that needs more than this has a design problem, not a limit problem.
    """
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)
    thread = threading.Thread(target=_watchdog, args=(host_bytes, 1.0), daemon=True)
    thread.start()


def clamp_envs(num_envs: int, training: bool) -> int:
    """Return the env count clipped to the project cap."""
    cap = MAX_TRAIN_ENVS if training else MAX_EVAL_ENVS
    return min(num_envs, cap)
