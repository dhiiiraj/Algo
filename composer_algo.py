#!/usr/bin/env python3
"""Entry point: multi-asset Composer-style rotation bot.

    python composer_algo.py --dry-run     # Paper account, log decision only
    python composer_algo.py               # Paper account, places orders
    python composer_algo.py --live        # REAL MONEY, places orders

Kept for existing schedules; same as `python run.py composer_rotation`.
"""

import sys

from algorithm import STRATEGIES
from engine.runner import main_for

if __name__ == "__main__":
    sys.exit(main_for(STRATEGIES["composer_rotation"]))
