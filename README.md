# How to Run This

This runs trading strategies against your Alpaca **paper trading** account
(fake money — safe to test with). Follow these steps exactly, in order.

## Every time you want to run it

1. Open the **Terminal** app.
2. Copy and paste this, then press Enter (this moves into the project folder):
   ```
   cd /Users/test/Desktop/algo/Algo
   ```
3. Copy and paste this, then press Enter (this turns on the project's tools):
   ```
   source .venv/bin/activate
   ```
   You'll see `(.venv)` appear at the start of the line — that means it worked.
4. To see what strategies are available:
   ```
   python run.py --list
   ```
5. To run one safely (checks what it *would* do, places no real orders):
   ```
   python run.py tqqq_weekly --dry-run
   ```
   or
   ```
   python run.py composer_rotation --dry-run
   ```

That's it — steps 2 and 3 are needed once per Terminal window; step 4/5 is
whatever you actually want to run.

## Where to see what happened

Every trade gets written to a file called **`trade_log.csv`** in this same
folder. Just double-click it to open it in Excel or Numbers — it's a normal
spreadsheet, always up to date, newest trades at the top.

## Things to know

- **`--dry-run`** means "just tell me what you'd do, don't actually trade."
  Always safe to run.
- Without `--dry-run`, it places real orders — but still only in the
  **paper (fake money)** account, unless `--live` is added.
- **Never run with `--live`** unless you've been told it's okay — that's
  real money.
- If it says `Market is CLOSED today`, that's normal on weekends/holidays —
  nothing is wrong, it just didn't do anything today.
- If something looks broken or confusing, stop and ask rather than guessing.
