"""Trading strategies: signal logic and target positions only.

To add an algorithm: copy algorithm/_template.py, fill it in, and register an
instance below. The key is the name used with `python run.py <name>`.
"""

from algorithm.base import Strategy
from algorithm.composer_rotation import ComposerRotation
from algorithm.tqqq_weekly import TqqqWeekly

STRATEGIES: dict[str, Strategy] = {
    "tqqq_weekly": TqqqWeekly(),
    "composer_rotation": ComposerRotation(),
}

_ids = [s.strategy_id for s in STRATEGIES.values()]
assert len(_ids) == len(set(_ids)), f"Duplicate strategy_id in registry: {_ids}"
