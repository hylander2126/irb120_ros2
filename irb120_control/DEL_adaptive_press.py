#!/usr/bin/env python3
"""Deprecated compatibility entry point for the unified arc_static behavior."""

import sys

from irb120_control.arc_static import main as arc_static_main


def main(args=None) -> int:
    """Preserve the former monitor-only command while using ArcStatic."""
    effective_args = list(sys.argv[1:] if args is None else args)
    if not any("object:=" in arg for arg in effective_args):
        effective_args.extend(["--ros-args", "-p", "object:=monitor"])
    return arc_static_main(args=effective_args)


if __name__ == "__main__":
    raise SystemExit(main())
