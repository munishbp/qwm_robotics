"""Memory guards that every script calls before it touches the GPU.

A nine day job shares this machine. These limits keep any script in this project from taking the
GPU or the host down. See docs/design.md section 2.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import torch

GPU_FRACTION = 0.25
GPU_BYTES = 8 * 1024**3
HOST_BYTES = 3 * 1024**3
MAX_TRAIN_ENVS = 512
MAX_EVAL_ENVS = 256


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _gpu_bytes_of_this_process() -> int:
    """GPU memory of this process as the driver reports it, which includes allocations that
    PyTorch's allocator does not see (for example MuJoCo Warp). Zero when nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    me = os.getpid()
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == me and parts[1].isdigit():
            total += int(parts[1]) * 1024**2
    return total


def _watchdog(host_bytes: int, gpu_bytes: int, period: float) -> None:
    tick = 0
    while True:
        time.sleep(period)
        tick += 1
        rss = _rss_bytes()
        if rss > host_bytes:
            sys.stderr.write(
                f"swarm.compute: host RSS {rss / 2**30:.2f} GB exceeds the cap "
                f"{host_bytes / 2**30:.2f} GB. The process exits now.\n"
            )
            sys.stderr.flush()
            os._exit(3)
        # nvidia-smi is slow, so the GPU check runs every few ticks.
        if tick % 3 == 0:
            gpu = _gpu_bytes_of_this_process()
            if gpu > gpu_bytes:
                sys.stderr.write(
                    f"swarm.compute: GPU memory {gpu / 2**30:.2f} GB exceeds the cap "
                    f"{gpu_bytes / 2**30:.2f} GB. The process exits now.\n"
                )
                sys.stderr.flush()
                os._exit(4)


def limit_memory(
    gpu_fraction: float = GPU_FRACTION, host_bytes: int = HOST_BYTES, gpu_bytes: int = GPU_BYTES
) -> None:
    """Cap GPU memory for this process and start a memory watchdog.

    The PyTorch cap makes an allocation past the fraction raise an error instead of taking memory
    from other processes. The watchdog exits the process when resident host memory or the
    driver reported GPU memory of this process passes its cap. The GPU check covers allocators
    PyTorch does not control, such as MuJoCo Warp. A script that needs more than this has a
    design problem, not a limit problem.
    """
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)
    thread = threading.Thread(target=_watchdog, args=(host_bytes, gpu_bytes, 1.0), daemon=True)
    thread.start()


def clamp_envs(num_envs: int, training: bool) -> int:
    """Return the env count clipped to the project cap."""
    cap = MAX_TRAIN_ENVS if training else MAX_EVAL_ENVS
    return min(num_envs, cap)
