# How to Run This

This runs trading strategies against your Alpaca **paper trading** account
(fake money — safe to test with). Follow these steps exactly, in order.

## First-time setup (only needed once, or on a new machine)

1. Open the **Terminal** app and move into the project folder:
   ```
   cd /Users/test/Desktop/algo/Algo
   ```
2. Create the project's virtual environment (a private folder of Python
   tools just for this project):
   ```
   python3 -m venv .venv
   ```
3. Turn it on:
   ```
   source .venv/bin/activate
   ```
4. Install everything the project needs:
   ```
   pip install -r requirements.txt
   ```
5. Create a file named `.env` in this same folder with your Alpaca paper
   trading keys:
   ```
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```
   (Get these from your Alpaca account's **paper trading** API keys page —
   never your live account keys.)

Once this is done, you don't need to repeat it — skip straight to "Every
time you want to run it" below from now on.

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
6. To actually place the trade (still paper/fake money, just a real order
   this time):
   ```
   python run.py tqqq_weekly
   ```
   or
   ```
   python run.py composer_rotation
   ```

That's it — steps 2 and 3 are needed once per Terminal window; step 4-6 is
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
