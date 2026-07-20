# Copyright (C) 2026 Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Plot results produced by the ``demo_affinity_test`` C++ executable.

Usage:
    ./demo_affinity_test | tee affinity.log
    python plot_affinity.py affinity.log [-o affinity.png]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Matches a data row, e.g.:
#        1     0.7434s       0.06     0.7447s       0.06     1.00x
ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)s\s+([\d.]+)\s+([\d.]+)s\s+([\d.]+)\s+([\d.]+)x\s*$")


def parse_log(text: str) -> dict[str, list[float]]:
    """Extract per-thread-count results from demo_affinity_test output.

    Args:
        text: Raw stdout captured from running the demo.

    Returns:
        Dict of equal-length lists: "threads", "unpinned_s", "unpinned_gbps",
        "pinned_s", "pinned_gbps", "speedup".
    """
    columns: dict[str, list[float]] = {
        "threads": [],
        "unpinned_s": [],
        "unpinned_gbps": [],
        "pinned_s": [],
        "pinned_gbps": [],
        "speedup": [],
    }
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match is None:
            continue
        threads, unpinned_s, unpinned_gbps, pinned_s, pinned_gbps, speedup = match.groups()
        columns["threads"].append(int(threads))
        columns["unpinned_s"].append(float(unpinned_s))
        columns["unpinned_gbps"].append(float(unpinned_gbps))
        columns["pinned_s"].append(float(pinned_s))
        columns["pinned_gbps"].append(float(pinned_gbps))
        columns["speedup"].append(float(speedup))

    if not columns["threads"]:
        raise ValueError("No data rows found in log -- wrong file, or demo failed?")

    return columns


def plot(data: dict[str, list[float]], title: str | None = None) -> Figure:
    """Build a two-panel figure: bandwidth vs threads, and speedup vs threads.

    Args:
        data: Parsed columns, as returned by parse_log.
        title: Optional suptitle (e.g. hostname / platform description).

    Returns:
        The created matplotlib Figure.
    """
    fig, (ax_bw, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_bw.plot(data["threads"], data["unpinned_gbps"], "o-", label="unpinned")
    ax_bw.plot(data["threads"], data["pinned_gbps"], "s-", label="pinned")
    ax_bw.set_xlabel("Threads")
    ax_bw.set_ylabel("Achieved bandwidth (GB/s)")
    ax_bw.set_title("compute_entities bandwidth")
    ax_bw.legend()
    ax_bw.grid(True, alpha=0.3)

    ax_speedup.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax_speedup.plot(data["threads"], data["speedup"], "o-", color="tab:green")
    ax_speedup.set_xlabel("Threads")
    ax_speedup.set_ylabel("Speedup (unpinned time / pinned time)")
    ax_speedup.set_title("Pinning speedup")
    ax_speedup.grid(True, alpha=0.3)

    for ax in (ax_bw, ax_speedup):
        ax.set_xticks(data["threads"])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    """Parse a demo_affinity_test log and write a plot to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        type=Path,
        nargs="?",
        help="Path to captured demo_affinity_test stdout. Reads stdin if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("affinity.png"),
        help="Output image path (default: affinity.png).",
    )
    parser.add_argument("--title", default=None, help="Optional plot title, e.g. machine name.")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively.")
    args = parser.parse_args()

    text = args.log.read_text() if args.log else sys.stdin.read()
    data = parse_log(text)

    fig = plot(data, title=args.title)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
