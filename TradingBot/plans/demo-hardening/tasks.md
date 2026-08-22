## Phase 0 — Safety fixes

- [x] Task 1: Wire realized day P&L into the circuit breaker in `app/core/engine.py` — derive `realized_pnl = balance − day_start_balance` from a fresh OANDA summary before each signal check and persist it to `account_state`, so the −3% day-loss guard can halt
  - Acceptance criteria: `_process_signal` fetches the OANDA account summary once before the breaker check, writes `realized_pnl` to `account_state`, and reuses the same summary for sizing at dispatch
  - Acceptance criteria: with a day-start baseline and a realized loss ≥ the limit, the breaker halts and the signal is blocked (engine test)
  - Acceptance criteria: breaker stays open when no day-start baseline exists
  - Scope in: engine breaker feed + Mongo upsert + tests
  - Scope out: order lifecycle / exit management

- [ ] Task 2: Free pending setups on DeepSeek fail-safe in `app/core/engine.py` — a `fail_safe:` verdict frees the setup key so the signal re-evaluates on the next candle; a genuine veto keeps it pending; record a new `OUTCOME_FAILSAFE`
  - Acceptance criteria: on a fail-safe decision the pending key is freed and the next closed candle re-emits the signal (engine test)
  - Acceptance criteria: on a genuine veto the key stays pending and later candles report `duplicate`
  - Acceptance criteria: `OUTCOME_FAILSAFE` returned for fail-safe verdicts
  - Scope in: engine veto branch + outcome constant + tests
  - Scope out: DeepSeek prompt changes or fail-safe defaults

- [ ] Task 3: Remove the redundant double `_pending.pop(key, None)` in `app/core/engine.py` filter branches — each blocked branch pops the key exactly once, no behavior change
  - Acceptance criteria: each filter-blocked branch pops the key once
  - Acceptance criteria: existing engine tests pass unchanged
  - Scope in: engine cleanup only
  - Scope out: any behavioral change

## Phase 1 — Cold-start correctness

- [ ] Task 4: Warm the H1 EMA(200) trend join at cold start — add a validated `backfill_count` setting (default 1000) used by the engine's OANDA backfill so a cold-started frame resamples to ≥ 200 H1 rows
  - Acceptance criteria: cold-start backfill requests ≥ 1000 M15 bars by default
  - Acceptance criteria: a 1000-bar frame resampled to H1 yields ≥ 200 rows (test)
  - Acceptance criteria: `backfill_count` configurable via env and validated > 0
  - Scope in: `app/config.py` setting + engine backfill + tests
  - Scope out: strategy or indicator changes

## Phase 2 — Demo observability & ops

- [ ] Task 5: Extend `/status` in `app/api/routes.py` — today's `signals`/`trade_logs` counts from Mongo plus a breaker snapshot (`day_loss_pct`, `trading_halted`); every Mongo-backed field degrades to `null`
  - Acceptance criteria: `/status` reports today's signal count, decision and order counts, and the breaker snapshot
  - Acceptance criteria: a downed Mongo still returns 200 with `null` fields
  - Scope in: routes + engine status + tests
  - Scope out: auth, dashboards, new endpoints

- [ ] Task 6: Practice-account demo runbook + local Mongo — add `docker-compose.yml` (mongodb service) and a README "Demo run (practice)" section covering `.env` setup, boot, verifying `/health` and `/status`, watching `trade_logs`, and confirming orders in the OANDA demo platform
  - Acceptance criteria: `docker-compose up` starts a local mongodb usable via `MONGODB_URI`
  - Acceptance criteria: README documents the practice-account demo run end to end
  - Scope in: `docker-compose.yml` + README demo section
  - Scope out: live-account docs, render.yaml changes, deployment

- [ ] Task 7: Persist `trading_halted=True` when the circuit breaker halts in `app/core/engine.py` — the flag is only ever written False (scheduler reset), so an intraday balance recovery re-opens the breaker, violating the "halts for the rest of the day" contract [bug]
