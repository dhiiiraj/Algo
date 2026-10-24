"""Bot lifecycle shared by every strategy.

args -> logging -> holiday guard -> credentials -> clients
     -> strategy.decide() -> execute_decision() -> email report
"""

import argparse
import logging
from datetime import datetime
from typing import Optional

from algorithm.base import MarketContext, Strategy
from alpaca_broker.client import create_clients, get_credentials, get_feed
from common.errors import StrategyError
from common.logging_setup import setup_logging
from common.market_calendar import ET, is_market_open_today
from common.notifications import send_email_report
from engine.executor import execute_decision, get_active_positions
from google_sheets import GoogleSheetExecutionLogger

log = logging.getLogger("algo.runner")


def add_mode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--live",
        action="store_true",
        help="Trade the LIVE account (real money). Default is paper.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log the decision but place no orders.",
    )


def run_strategy(strategy: Strategy, live: bool, dry_run: bool) -> int:
    setup_logging(strategy.get_log_file_name())
    sid = strategy.strategy_id

    # ---- MARKET HOLIDAY GUARD ----
    if not is_market_open_today():
        today_stamp = datetime.now(ET).strftime("%Y-%m-%d")
        log.info(
            "Market is CLOSED today (%s). Exiting job safely with success code.",
            today_stamp,
        )
        return 0

    exec_logger = GoogleSheetExecutionLogger(sid, strategy.dust_threshold)
    mode = "PAPER" if not live else "*** LIVE ***"
    run_stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")

    api_key, secret_key = get_credentials()
    if not api_key or not secret_key:
        log.error("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.")
        send_email_report(
            f"[{sid}] ERROR ({run_stamp})",
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.",
        )
        return 1

    log.info(
        "Starting Strategy [%s] | Target Qty=%s shares | mode=%s | dry_run=%s",
        sid,
        strategy.target_qty(),
        mode,
        dry_run,
    )

    try:
        feed = get_feed()
    except ValueError:
        log.error("Invalid ALPACA_FEED (use 'iex' or 'sip').")
        send_email_report(
            f"[{sid}] ERROR ({run_stamp})",
            "Invalid ALPACA_FEED env var (use 'sip' or 'iex').",
        )
        return 1

    data_client, trading_client = create_clients(api_key, secret_key, paper=not live)

    try:
        positions = get_active_positions(strategy, exec_logger)
        decision = strategy.decide(
            MarketContext(data_client=data_client, feed=feed, positions=positions)
        )
        outcome = execute_decision(
            strategy,
            decision,
            trading_client,
            exec_logger,
            dry_run,
            active_positions=positions,
        )

        body = (
            f"Run time: {run_stamp}\n"
            f"Strategy ID: {sid}\n"
            f"Target Share Quantity: {strategy.target_qty()} shares\n"
            f"Mode: {mode}{' (dry-run)' if dry_run else ''}\n\n"
            f"{decision.email_details}\n\n"
            f"Reason: {decision.reason}\n\n"
            f"Execution outcome:\n  {outcome}\n"
        )
        send_email_report(f"[{sid}] {decision.email_subject}", body)
        return 0

    except StrategyError as e:
        log.error("ABORTED: %s", e)
        send_email_report(
            f"[{sid}] ABORTED ({run_stamp})",
            f"Run time: {run_stamp}\nMode: {mode}\n\nABORTED: {e}",
        )
        return 1
    except Exception as e:
        log.exception("Unhandled error")
        send_email_report(
            f"[{sid}] UNHANDLED ERROR ({run_stamp})",
            f"Run time: {run_stamp}\nMode: {mode}\n\nUnhandled error: {e}",
        )
        return 1


def main_for(strategy: Strategy, argv: Optional[list] = None) -> int:
    """Entry point for a single, fixed strategy (used by main.py / composer_algo.py)."""
    p = argparse.ArgumentParser(description=strategy.description)
    add_mode_args(p)
    args = p.parse_args(argv)
    return run_strategy(strategy, live=args.live, dry_run=args.dry_run)
