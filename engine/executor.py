"""Shared trade executor: moves a strategy from its current positions to its target.

Rules (same for every strategy):
    * Positions come from the Google Sheet, tagged by strategy_id, so several
      strategies can share one Alpaca account without touching each other.
    * If this strategy already has open orders, do nothing (avoid duplicates).
    * Sell every held symbol that is not in the target.
    * Buy every target symbol that is not held. Symbols already held are kept
      as-is (never topped up).
    * Nothing to do -> write a HOLD row.
"""

import logging
import uuid
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

from algorithm.base import Decision, Strategy
from alpaca_broker.orders import (
    assert_account_can_trade,
    assert_asset_tradable,
    assert_order_ok,
    get_strategy_open_orders,
    is_market_open_now,
    order_status,
    submit_market_order,
    wait_for_order,
)
from google_sheets import GoogleSheetExecutionLogger

log = logging.getLogger("algo.executor")


def get_active_positions(
    strategy: Strategy, exec_logger: GoogleSheetExecutionLogger
) -> dict[str, float]:
    positions = exec_logger.get_all_strategy_positions(strategy.universe)
    return {s: q for s, q in positions.items() if q > strategy.dust_threshold}


def _trade(
    strategy: Strategy,
    trading_client: TradingClient,
    exec_logger: GoogleSheetExecutionLogger,
    symbol: str,
    qty: float,
    side: OrderSide,
    reason: str,
    dry_run: bool,
    market_open: bool,
) -> str:
    action = "BUY" if side == OrderSide.BUY else "SELL"
    client_order_id = f"{strategy.strategy_id}_{action}_{uuid.uuid4().hex[:8]}"

    if dry_run:
        msg = f"[DRY RUN] Would {action} {qty} shares of {symbol}."
        log.info("%s | Client Order ID: %s", msg, client_order_id)
        exec_logger.log_execution(
            broker_order_id=client_order_id,
            symbol=symbol,
            action=action,
            notional_usd=0.0,
            qty_shares=qty,
            status="DRY_RUN",
            dry_run=True,
            reason=reason,
        )
        return msg

    order = submit_market_order(trading_client, symbol, qty, side, client_order_id)
    log.info(
        "Submitted %s %s shares of %s | client_id=%s | order_id=%s | status=%s",
        action,
        qty,
        symbol,
        client_order_id,
        order.id,
        order_status(order),
    )

    if market_open:
        order = wait_for_order(trading_client, order)
        log.info(
            "%s order status: %s (filled_qty=%s)",
            action,
            order_status(order),
            order.filled_qty,
        )

    assert_order_ok(order, f"{action} {symbol}")

    filled_qty = float(order.filled_qty or qty)
    filled_price = float(order.filled_avg_price or 0.0)
    est_notional = round(filled_qty * filled_price, 2) if filled_price else 0.0

    exec_logger.log_execution(
        broker_order_id=str(order.id),
        symbol=symbol,
        action=action,
        notional_usd=est_notional,
        qty_shares=filled_qty,
        status=order_status(order),
        dry_run=False,
        reason=reason,
    )
    return (
        f"{action} {filled_qty} shares of {symbol} (~${est_notional:.2f}) |"
        f" Order ID: {order.id} | Status: {order_status(order)}"
    )


def execute_decision(
    strategy: Strategy,
    decision: Decision,
    trading_client: TradingClient,
    exec_logger: GoogleSheetExecutionLogger,
    dry_run: bool,
    active_positions: Optional[dict[str, float]] = None,
) -> str:
    log.info(
        "================ EXECUTION [Strategy ID: %s] ================",
        strategy.strategy_id,
    )
    unknown = set(decision.target) - set(strategy.universe)
    if unknown:
        raise ValueError(f"Target symbols {sorted(unknown)} are not in the strategy universe.")

    assert_account_can_trade(trading_client)
    market_open = is_market_open_now(trading_client)

    if active_positions is None:
        active_positions = get_active_positions(strategy, exec_logger)
    log.info(
        "Strategy [%s] Tracked Positions: %s | Target: %s",
        strategy.strategy_id,
        active_positions,
        decision.target,
    )

    strat_open_orders = get_strategy_open_orders(
        trading_client, strategy.universe, strategy.strategy_id
    )
    if strat_open_orders:
        msg = (
            f"SKIP: {len(strat_open_orders)} open order(s) already pending for "
            f"strategy {strategy.strategy_id}; skipping to avoid duplicates."
        )
        log.warning(msg)
        return msg

    to_sell = [s for s in active_positions if s not in decision.target]
    to_buy = [s for s in decision.target if s not in active_positions]

    if not to_sell and not to_buy:
        if decision.target:
            symbol = next(iter(decision.target))
            msg = (
                f"NO ACTION: already holding target {symbol}"
                f" ({active_positions[symbol]} shares)."
            )
        else:
            symbol = strategy.universe[0]
            msg = "NO ACTION: target is FLAT and strategy holds nothing."
        log.info(msg)
        exec_logger.log_execution(
            broker_order_id=f"HOLD_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            action="HOLD",
            notional_usd=0.0,
            qty_shares=active_positions.get(symbol, 0.0),
            status="NO_ACTION",
            dry_run=dry_run,
            reason=decision.reason,
        )
        return msg

    outcomes = []
    for symbol in to_sell:
        qty = active_positions[symbol]
        log.info("SELL: exiting %s (%s shares).", symbol, qty)
        outcomes.append(
            _trade(
                strategy,
                trading_client,
                exec_logger,
                symbol,
                qty,
                OrderSide.SELL,
                f"Exit {symbol}: {decision.reason}",
                dry_run,
                market_open,
            )
        )

    for symbol in to_buy:
        qty = round(decision.target[symbol], 4)
        if qty < strategy.min_trade_qty:
            msg = (
                f"Trade quantity {qty} for {symbol} below minimum threshold"
                f" {strategy.min_trade_qty}; not buying."
            )
            log.error(msg)
            outcomes.append(msg)
            continue
        assert_asset_tradable(trading_client, symbol)
        log.info("BUY: entering %s (%s shares).", symbol, qty)
        outcomes.append(
            _trade(
                strategy,
                trading_client,
                exec_logger,
                symbol,
                qty,
                OrderSide.BUY,
                decision.reason,
                dry_run,
                market_open,
            )
        )

    return "\n  ".join(outcomes)
