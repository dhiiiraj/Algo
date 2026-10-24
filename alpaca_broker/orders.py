import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from common.errors import StrategyError

log = logging.getLogger("algo.alpaca.orders")

ORDER_POLL_TIMEOUT_SEC = 20

TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
    OrderStatus.DONE_FOR_DAY,
}
FAILED_STATUSES = {
    OrderStatus.REJECTED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
}


def order_status(order) -> str:
    return str(getattr(order.status, "value", order.status))


def wait_for_order(client: TradingClient, order):
    latest = order
    deadline = time.monotonic() + ORDER_POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        latest = client.get_order_by_id(order.id)
        if latest.status in TERMINAL_STATUSES:
            break
        time.sleep(2)

    if latest.status not in TERMINAL_STATUSES:
        log.warning(
            "Order %s did not reach a terminal state within %ds (status: %s).",
            order.id,
            ORDER_POLL_TIMEOUT_SEC,
            order_status(latest),
        )
    return latest


def assert_order_ok(order, label: str) -> None:
    if order.status in FAILED_STATUSES:
        raise StrategyError(f"{label} order {order.id} ended as {order_status(order)}")


def assert_account_can_trade(client: TradingClient) -> None:
    account = client.get_account()
    if account.trading_blocked or account.account_blocked:
        raise StrategyError("Account is blocked from trading; aborting.")


def assert_asset_tradable(client: TradingClient, symbol: str) -> None:
    if not client.get_asset(symbol).tradable:
        raise StrategyError(f"{symbol} is not tradable on this account.")


def is_market_open_now(client: TradingClient) -> bool:
    """Live clock check; day orders placed while closed queue for the next open."""
    market_open = True
    try:
        market_open = bool(client.get_clock().is_open)
    except Exception as e:
        log.warning("Could not check market clock: %s", e)
    if not market_open:
        log.warning("Market is CLOSED. Day orders will queue for the next open.")
    return market_open


def get_strategy_open_orders(
    client: TradingClient, symbols: list[str], strategy_id: str
) -> list:
    """Open orders whose client_order_id is prefixed with this strategy's ID."""
    open_orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols)
    )
    return [o for o in open_orders if str(o.client_order_id).startswith(strategy_id)]


def submit_market_order(
    client: TradingClient,
    symbol: str,
    qty: float,
    side: OrderSide,
    client_order_id: str,
):
    return client.submit_order(
        MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
    )
