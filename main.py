#!/usr/bin/env python3
"""Entry point: weekly QQQ-signal / TQQQ-trade bot.

    python main.py --dry-run     # Paper account, log decision only
    python main.py               # Paper account, places orders
    python main.py --live        # REAL MONEY, places orders

Kept for existing schedules; same as `python run.py tqqq_weekly`.
"""

import sys

from algorithm import STRATEGIES
from engine.runner import main_for

if __name__ == "__main__":
    sys.exit(main_for(STRATEGIES["tqqq_weekly"]))
