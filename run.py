#!/usr/bin/env python3
"""Run any registered algorithm.

    python run.py --list                         # show available algorithms
    python run.py tqqq_weekly --dry-run          # paper account, no orders
    python run.py composer_rotation              # paper account, places orders
    python run.py composer_rotation --live       # REAL MONEY

Algorithms are registered in algorithm/__init__.py.
"""

import argparse
import sys

from algorithm import STRATEGIES
from engine.runner import add_mode_args, run_strategy


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run a trading algorithm.")
    p.add_argument("strategy", nargs="?", choices=sorted(STRATEGIES), help="Algorithm to run.")
    p.add_argument("--list", action="store_true", help="List available algorithms.")
    add_mode_args(p)
    args = p.parse_args(argv)

    if args.list or not args.strategy:
        for name, s in sorted(STRATEGIES.items()):
            print(f"{name:20s} {s.strategy_id:18s} qty env: {s.qty_env_var:22s} {s.description}")
        return 0 if args.list else 2

    return run_strategy(STRATEGIES[args.strategy], live=args.live, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
