"""Real-time per-link body position tracking error plot for sim2sim evaluation.

Runs a matplotlib figure in a separate daemon process so it never blocks the
200 Hz MuJoCo physics loop.  The main process calls push() once per step-sync
frame; the subprocess drains the queue and redraws at the configured Hz.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time

import numpy as np


def _plot_worker(
    q: mp.Queue,
    link_names: list[str],
    ymax_mm: float,
    refresh_hz: float,
) -> None:
    """Entry point for the subprocess.  Maintains full per-link error history."""
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("Agg")  # headless fallback — window won't show

    import matplotlib.pyplot as plt

    n = len(link_names)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
    )
    fig.suptitle("Sim2Sim Link Tracking Error (mm)", fontsize=11)

    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    x_data: list[list[float]] = [[] for _ in range(n)]
    y_data: list[list[float]] = [[] for _ in range(n)]

    lines = []
    for i, (ax, name) in enumerate(zip(axes_flat[:n], link_names)):
        ax.set_title(name, fontsize=8, pad=3)
        ax.set_xlabel("frame", fontsize=7)
        ax.set_ylabel("error (mm)", fontsize=7)
        ax.set_ylim(0, ymax_mm)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        # Reference lines at 30 mm and 100 mm
        ax.axhline(30, color="orange", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.axhline(100, color="red", linewidth=0.6, linestyle="--", alpha=0.7)
        (ln,) = ax.plot([], [], lw=1.2, color="tab:blue")
        lines.append(ln)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.ion()
    plt.show()

    sleep_time = 1.0 / max(refresh_hz, 0.5)

    while True:
        # Drain all pending items without blocking
        new_items: list[tuple[int, list[float]]] = []
        while True:
            try:
                new_items.append(q.get_nowait())
            except queue.Empty:
                break

        if new_items:
            for frame_idx, errors in new_items:
                for i in range(n):
                    x_data[i].append(frame_idx)
                    y_data[i].append(errors[i])

            x_max = max(x_data[0]) if x_data[0] else 10

            for i, (ln, ax) in enumerate(zip(lines, axes_flat[:n])):
                ln.set_data(x_data[i], y_data[i])
                ax.set_xlim(0, x_max + max(int(x_max * 0.05), 10))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        time.sleep(sleep_time)


class Sim2SimLinkErrorPlot:
    """Manages a per-link error plot subprocess.

    Usage::

        plot = Sim2SimLinkErrorPlot(["pelvis", "left_ankle_roll_link"], ymax_mm=300)
        plot.start()
        ...
        plot.push(frame_idx, errors_mm_array)   # non-blocking, call every step
        ...
        plot.close()
    """

    def __init__(
        self,
        link_names: list[str],
        ymax_mm: float = 300.0,
        refresh_hz: float = 20.0,
    ) -> None:
        self._link_names = list(link_names)
        self._ymax_mm = ymax_mm
        self._refresh_hz = refresh_hz
        # maxsize keeps memory bounded; 2000 frames >> any realistic pkl length
        self._queue: mp.Queue = mp.Queue(maxsize=2000)
        self._proc: mp.Process | None = None

    def start(self) -> None:
        self._proc = mp.Process(
            target=_plot_worker,
            args=(self._queue, self._link_names, self._ymax_mm, self._refresh_hz),
            daemon=True,
        )
        self._proc.start()

    def push(self, frame_idx: int, errors_mm: np.ndarray) -> None:
        """Push one step-sync data point.  Never blocks — drops frame if queue full."""
        try:
            self._queue.put_nowait((frame_idx, errors_mm.tolist()))
        except Exception:
            pass

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._proc = None
