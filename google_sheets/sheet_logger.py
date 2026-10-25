import base64
import json
import logging
import os
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

log = logging.getLogger("algo.google_sheets")

DEFAULT_DUST_THRESHOLD = 0.001


class GoogleSheetExecutionLogger:
    """Logs trade executions and maintains strategy-level virtual positions.

    Each row is tagged with ``strategy_id`` so several strategies can share one
    sheet and still compute their own positions independently.
    """

    def __init__(
        self, strategy_id: str, dust_threshold: float = DEFAULT_DUST_THRESHOLD
    ):
        self.strategy_id = strategy_id
        self.dust_threshold = dust_threshold
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64")
        self.worksheet = None

        if not self.sheet_id or not self.creds_b64:
            log.warning(
                "GOOGLE_SHEET_ID or GOOGLE_CREDENTIALS_B64 not set. Google"
                " Sheets logging disabled."
            )
            return

        try:
            raw_b64 = self.creds_b64.strip()

            if (raw_b64.startswith('"') and raw_b64.endswith('"')) or (
                raw_b64.startswith("'") and raw_b64.endswith("'")
            ):
                raw_b64 = raw_b64[1:-1].strip()

            missing_padding = len(raw_b64) % 4
            if missing_padding:
                raw_b64 += "=" * (4 - missing_padding)

            decoded_json = base64.b64decode(raw_b64).decode("utf-8").strip()
            creds_dict = json.loads(decoded_json)

            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace(
                    "\\n", "\n"
                )

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )
            client = gspread.authorize(creds)
            self.worksheet = client.open_by_key(self.sheet_id).sheet1
            self._ensure_headers()
        except Exception as e:
            log.error("Failed to connect to Google Sheets: %s", e)
            self.worksheet = None

    def _ensure_headers(self):
        try:
            if not self.worksheet.get_all_values():
                headers = [
                    "timestamp_utc",
                    "strategy_id",
                    "broker_order_id",
                    "symbol",
                    "action",
                    "notional_usd",
                    "qty_shares",
                    "status",
                    "dry_run",
                    "reason",
                ]
                self.worksheet.append_row(headers)
        except Exception as e:
            log.warning("Could not verify or write Google Sheet headers: %s", e)

    def log_execution(
        self,
        broker_order_id: str,
        symbol: str,
        action: str,
        notional_usd: float,
        qty_shares: float,
        status: str,
        dry_run: bool,
        reason: str,
    ):
        if not self.worksheet:
            log.warning("Skipping Google Sheets log entry (client not connected).")
            return

        try:
            row = [
                datetime.now(timezone.utc).isoformat(),
                self.strategy_id,
                broker_order_id,
                symbol,
                action.upper(),
                round(notional_usd, 2) if notional_usd else 0.0,
                round(qty_shares, 4) if qty_shares else 0.0,
                str(status).upper(),
                dry_run,
                reason,
            ]
            self.worksheet.append_row(row)
            log.info(
                "Successfully logged trade execution [Strategy: %s | OrderID: %s] to"
                " Google Sheets.",
                self.strategy_id,
                broker_order_id,
            )
        except Exception as e:
            log.error("Failed to append trade record to Google Sheet: %s", e)

    def get_strategy_position(self, symbol: str) -> float:
        """Calculates net position for THIS strategy using filled logs from Google Sheets."""
        if not self.worksheet:
            log.warning(
                "Sheet logging disabled; cannot deduce virtual position. Returning 0.0."
            )
            return 0.0

        try:
            records = self.worksheet.get_all_records()
            net_qty = 0.0
            for r in records:
                row_strat = str(r.get("strategy_id", "")).strip()
                row_symbol = str(r.get("symbol", "")).strip()
                row_status = str(r.get("status", "")).strip().upper()
                row_action = str(r.get("action", "")).strip().upper()

                # Flexible status match for any filled/done string variations
                is_filled = "FILLED" in row_status or row_status in {"DONE", "EXECUTED"}

                if (
                    row_strat == self.strategy_id
                    and row_symbol == symbol
                    and is_filled
                ):
                    try:
                        shares = float(r.get("qty_shares", 0.0))
                    except (ValueError, TypeError):
                        shares = 0.0

                    if row_action == "BUY":
                        net_qty += shares
                    elif row_action == "SELL":
                        net_qty -= shares

            if 0 < net_qty < self.dust_threshold:
                return 0.0
            return max(net_qty, 0.0)

        except Exception as e:
            log.error("Error reading virtual position from Google Sheets: %s", e)
            return 0.0

    def get_all_strategy_positions(self, tickers: list[str]) -> dict[str, float]:
        """Returns virtual positions for all strategy tickers."""
        return {t: self.get_strategy_position(t) for t in tickers}
