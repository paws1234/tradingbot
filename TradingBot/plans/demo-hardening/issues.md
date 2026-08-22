# Issue mapping — demo-hardening

Host: GitHub | Repo: paws1234/tradingbot | Default branch: main

> This mapping is created at `/plan` time (one issue per task) and updated by
> `/implement`, `/review`, `/test`, and `/amend` (see CLAUDE.md §Git Host
> Integration). `[bug]` rows are out-of-scope issues captured automatically.
> Status reflects the task state in `tasks.md` and the issue state on the host.

| Task | Issue | Status | Title |
|------|-------|--------|-------|
| Task 1 | #55 | closed | Wire realized day P&L into the circuit breaker in `app/core/engine.py` — derive `realized_pnl = balance − day_start_balance` from a fresh OANDA summary before each signal check and persist it to `account_state`, so the −3% day-loss guard can halt |
| Task 2 | #56 | closed | Free pending setups on DeepSeek fail-safe in `app/core/engine.py` — a `fail_safe:` verdict frees the setup key so the signal re-evaluates on the next candle; a genuine veto keeps it pending; record a new `OUTCOME_FAILSAFE` |
| Task 3 | #57 | closed | Remove the redundant double `_pending.pop(key, None)` in `app/core/engine.py` filter branches — each blocked branch pops the key exactly once, no behavior change |
| Task 4 | #58 | open | Warm the H1 EMA(200) trend join at cold start — add a validated `backfill_count` setting (default 1000) used by the engine's OANDA backfill so a cold-started frame resamples to ≥ 200 H1 rows |
| Task 5 | #59 | open | Extend `/status` in `app/api/routes.py` — today's `signals`/`trade_logs` counts from Mongo plus a breaker snapshot (`day_loss_pct`, `trading_halted`); every Mongo-backed field degrades to `null` |
| Task 6 | #60 | open | Practice-account demo runbook + local Mongo — add `docker-compose.yml` (mongodb service) and a README "Demo run (practice)" section covering `.env` setup, boot, verifying `/health` and `/status`, watching `trade_logs`, and confirming orders in the OANDA demo platform |
| Bug 1 | #62 | open | Persist `trading_halted=True` when the circuit breaker halts |
