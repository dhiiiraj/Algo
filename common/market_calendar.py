import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import exchange_calendars as xcal

ET = ZoneInfo("America/New_York")

log = logging.getLogger("algo.market_calendar")


def is_market_open_today() -> bool:
    """Checks if NYSE/NASDAQ is scheduled for an active trading session today."""
    try:
        nyse = xcal.get_calendar("XNYS")
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        return nyse.is_session(today_str)
    except Exception as e:
        log.warning("Could not check market calendar (%s); proceeding as open.", e)
        return True
