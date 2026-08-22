approved: true
# Plan: demo-hardening — Make the trading bot demo-ready on an OANDA practice account

> Builds on the `trading-bot` plan (Tasks 1–27, merged or in PR). This plan hardens the live
> pipeline so the operator can run it end-to-end against an OANDA *practice* account and
> validate signals → filters → veto → orders on the broker's demo platform.

## Goal

Make the trading bot safe, observably correct, and runnable end-to-end against an OANDA
practice (demo) account so the live workflow can be validated on the broker platform.

## Scope

In:
- Wire real realized day P&L into the circuit breaker so the −3% day-loss guard actually halts
- Free pending setups on a DeepSeek fail-safe (gate unavailable) so signals re-evaluate on the next candle; genuine vetoes keep blocking re-fire
- Remove the redundant double `_pending.pop(key, None)` in the engine's filter branches
- Warm the H1 EMA(200) trend join at cold start via a deeper, configurable backfill
- Demo observability: extend `/status` with today's signal/decision/order counts and a breaker snapshot
- Practice-account demo runbook + local MongoDB via `docker-compose.yml`
- Tests for every change; the existing suite stays green

Out:
- Live (non-practice) trading / real money
- Order lifecycle beyond entry (trailing, cancel/modify, exit management)
- Backtesting / historical optimization
- New strategies, indicator changes, or filter-parameter tuning
- Web UI / dashboard beyond `/status`
- Auth or write endpoints on the API

## Architecture

Pipeline unchanged (daily context → streams → signals → filters → veto → sizing → dispatch).
The changes are surgical:

- **Circuit breaker feed** — at each signal check the engine fetches a fresh OANDA account
  summary (reusing it for sizing later in the same path), derives
  `realized_pnl ≈ balance − day_start_balance` (OANDA `balance` excludes unrealized P&L), and
  persists it to `account_state`. The scheduler keeps resetting the day-start baseline at 00:00 UTC,
  so breaker math and `/status` share one source of truth.
- **Veto / fail-safe semantics** — the DeepSeek layer already stamps unavailable-verdicts with a
  `fail_safe:` reason. The engine treats those as non-terminal (frees the pending setup key, so the
  signal re-fires on the next closed candle once the gate recovers) and genuine vetoes as terminal
  (key stays pending until invalidated or day rollover). New outcome `OUTCOME_FAILSAFE`.
- **Cold start** — backfill depth becomes a validated setting `backfill_count` (default 1000 M15
  bars ≈ 250 H1 rows), so the H1 EMA(200) trend join is seeded from real history rather than a
  single close.
- **Observability** — `/status` adds today's Mongo counts (`signals`, `trade_logs` decisions and
  orders) and a computed breaker snapshot (`day_loss_pct`, `trading_halted`); every Mongo-backed
  field still degrades to `null` so the endpoint survives a downed DB.
- **Ops** — `docker-compose.yml` runs a local `mongodb` service the app can use via
  `MONGODB_URI`; README gains a "Demo run (practice)" section.

CI already exists (`.github/workflows/ci.yml`) — no new pipeline work.

## Phases

- Phase 0 — Safety fixes (breaker feed, fail-safe semantics, cleanup)
- Phase 1 — Cold-start correctness (backfill warm-up)
- Phase 2 — Demo observability & ops (/status, demo runbook, local Mongo)
