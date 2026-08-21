## Phase 0 — Scaffolding & configuration

- [x] Task 1: Scaffold the `TradingBot/` project — app package, `tests/`, `requirements.txt`, `.env.example`, `.dockerignore`, README stub
  - Acceptance criteria: `TradingBot/` package imports cleanly with `app/` and `tests/` present
  - Acceptance criteria: `pytest` collects and runs from `TradingBot/`; `requirements.txt` installs without error
  - Acceptance criteria: `.env.example` and `.dockerignore` exist
  - Scope in: create `TradingBot/` layout — `app/`, `tests/`, `requirements.txt`, `.env.example`, `.dockerignore`, README stub
  - Scope in: minimal package wiring (`__init__.py`)
  - Scope out: any feature logic (indicators, strategies, data clients)
  - Scope out: real credentials or secrets

- [x] Task 2: Implement `app/config.py` — pydantic-settings env validation (OANDA practice/live switch, INSTRUMENTS + STRATEGIES parsing, risk thresholds, DeepSeek defaults)
  - Acceptance criteria: INSTRUMENTS + STRATEGIES parsed from comma-separated env into lists
  - Acceptance criteria: ACCOUNT_TYPE accepts only `practice`/`live`; missing/invalid fails fast
  - Acceptance criteria: risk thresholds and DeepSeek defaults present with validation
  - Scope in: pydantic-settings model + env loading/validation in `app/config.py`
  - Scope in: practice/live env switch
  - Scope out: data clients (OANDA/Finnhub/DeepSeek)
  - Scope out: strategy and indicator logic

- [x] Task 3: Define `app/models/schemas.py` — Candle, PriceTick, NewsItem, CalendarEvent, Signal, TradeDecision, OrderResult pydantic models
  - Acceptance criteria: all 7 models defined with correct fields/types
  - Acceptance criteria: models JSON-serialize round-trip
  - Scope in: pydantic model definitions + field validation
  - Scope out: DB persistence / Mongo
  - Scope out: API routes / engine

## Phase 1 — Data layer

- [x] Task 4: Implement `app/data/mongo.py` — motor client; collections `daily_context`, `account_state`, `trade_logs`, `signals`; upsert helpers
  - Acceptance criteria: motor AsyncIOMotorClient wired; 4 collections created
  - Acceptance criteria: upsert helpers write/read correctly
  - Acceptance criteria: missing MONGODB_URL fails fast with a clear message
  - Scope in: mongo client + collection access
  - Scope in: upsert/CRUD helpers
  - Scope out: trading/strategy logic
  - Scope out: stream consumers

- [x] Task 5: Implement `app/data/oanda.py` — REST (candles, balance, orders) + NDJSON pricing stream with auto-reconnect
  - Acceptance criteria: REST calls for candles, balance, orders are correct and authenticated
  - Acceptance criteria: pricing stream parses PRICE vs HEARTBEAT lines
  - Acceptance criteria: stream auto-reconnects with backoff on drop
  - Scope in: OANDA v20 REST client
  - Scope in: NDJSON pricing stream + reconnect
  - Scope out: order strategy / sizing
  - Scope out: account setup or credential handling beyond the client

- [x] Task 6: Implement `app/data/finnhub.py` — news WebSocket with REST polling fallback
  - Acceptance criteria: news delivered over WebSocket
  - Acceptance criteria: REST polling fallback when the WebSocket fails
  - Acceptance criteria: reconnects on drop
  - Scope in: Finnhub news WebSocket client
  - Scope in: REST polling fallback
  - Scope out: news content processing / sentiment
  - Scope out: scheduling

- [x] Task 7: Implement `app/data/forexfactory.py` — calendar scraper (httpx + bs4, EST→UTC, retries, parse tolerance, manual override)
  - Acceptance criteria: events parsed with EST→UTC conversion
  - Acceptance criteria: malformed rows tolerated (no crash)
  - Acceptance criteria: retries on request failure; manual override hook exists
  - Scope in: ForexFactory calendar scraper (httpx + bs4)
  - Scope in: retry + manual override mechanism
  - Scope out: blackout-window logic
  - Scope out: scheduler

## Phase 2 — Indicators & strategies

- [x] Task 8: Implement `app/indicators/technical.py` — ema, sma, rsi, true_range, atr, bollinger, adx, donchian, fvg, session_range, m15→h1 resample
  - Acceptance criteria: indicators match known hand-computed series
  - Acceptance criteria: m15→h1 resample produces correct bars
  - Acceptance criteria: no lookahead — rolling windows shifted 1 bar
  - Scope in: pure indicator functions
  - Scope in: m15→h1 resample
  - Scope out: signal generation
  - Scope out: strategy decisions

- [x] Task 9: Implement `app/strategy/signals.py` — four strategy signal functions + STRATEGY_REGISTRY + dedup
  - Acceptance criteria: all 4 strategy signal functions present
  - Acceptance criteria: STRATEGY_REGISTRY dispatches each strategy
  - Acceptance criteria: dedup per (strategy, instrument, day, side)
  - Scope in: four signal functions
  - Scope in: registry + dedup
  - Scope out: filters (breaker/blackout)
  - Scope out: sizing / orders

- [x] Task 10: Implement `app/strategy/filters.py` — circuit breaker halting at day loss ≤ −3%, and ±30 min news blackout
  - Acceptance criteria: breaker halts trading at day loss ≤ −3%
  - Acceptance criteria: news blackout blocks entries ±30 min around high-impact events
  - Scope in: circuit breaker filter
  - Scope in: news blackout filter
  - Scope out: technical filters (EMA/RSI/ATR)
  - Scope out: exits / trailing, other strategies

- [x] Task 11: Implement `app/strategy/sizing.py` — 1% risk sizing, OANDA unit rounding, SL/TP, MARKET payload
  - Acceptance criteria: units = (balance × 0.01) / |entry − SL| computed
  - Acceptance criteria: units rounded to OANDA lot precision
  - Acceptance criteria: MARKET payload includes stopLossOnFill/takeProfitOnFill
  - Scope in: 1% risk sizing calculation
  - Scope in: order payload builder
  - Scope out: order execution / dispatch
  - Scope out: lifecycle / trailing management

## Phase 3 — DeepSeek gate

- [x] Task 12: Implement `app/data/deepseek.py` — openai SDK wrapper, JSON response_format, retry/backoff, parse fail-safe `execute=false`
  - Acceptance criteria: openai SDK wrapper with DeepSeek base_url + JSON response_format
  - Acceptance criteria: retry/backoff on transient errors
  - Acceptance criteria: parse failure returns `execute=false` (fail-safe)
  - Scope in: DeepSeek client wrapper
  - Scope in: JSON veto gate parsing
  - Scope out: other LLMs / models
  - Scope out: prompt tuning beyond spec

- [x] Task 13: Audit-log DeepSeek decisions + order results to `trade_logs`
  - Acceptance criteria: every DeepSeek decision written to `trade_logs`
  - Acceptance criteria: order results (fills/rejects) written to `trade_logs`
  - Scope in: `trade_logs` audit writes
  - Scope out: analytics / dashboards / reporting
  - Scope out: UI

## Phase 4 — Engine & FastAPI

- [x] Task 14: Implement `app/core/scheduler.py` — AsyncIOScheduler daily 00:00 UTC context build + `account_state` reset
  - Acceptance criteria: AsyncIOScheduler job scheduled daily 00:00 UTC
  - Acceptance criteria: job builds daily context and resets `account_state`
  - Scope in: scheduler setup + daily job
  - Scope out: stream lifecycle
  - Scope out: engine orchestration

- [x] Task 15: Implement `app/core/engine.py` — per-instrument stream tasks, candle builder, signals → filters → DeepSeek → dispatch
  - Acceptance criteria: one stream task per instrument started/cancelled
  - Acceptance criteria: candles built from ticks
  - Acceptance criteria: pipeline order signals → filters → DeepSeek → dispatch enforced
  - Scope in: engine orchestration
  - Scope in: candle builder
  - Scope out: API routes
  - Scope out: deployment / scheduler

- [x] Task 16: Implement `app/api/routes.py` + `app/main.py` — `/health`, `/status`, lifespan start/stop with graceful cancel
  - Acceptance criteria: `/health` returns 200
  - Acceptance criteria: `/status` reports current state
  - Acceptance criteria: lifespan starts/stops engine with graceful cancellation
  - Scope in: FastAPI app + `/health` + `/status`
  - Scope in: lifespan wiring
  - Scope out: auth / frontend
  - Scope out: metrics collection

## Phase 5 — Docker & deploy files

- [x] Task 17: Add `Dockerfile` — python:3.12-slim, non-root, EXPOSE, HEALTHCHECK, uvicorn CMD
  - Acceptance criteria: image builds successfully
  - Acceptance criteria: runs as non-root; EXPOSE + HEALTHCHECK defined
  - Acceptance criteria: `uvicorn` CMD serves the app
  - Scope in: Docker packaging (python:3.12-slim)
  - Scope out: orchestration / deploy config (render.yaml)
  - Scope out: app code changes

- [x] Task 18: Add `render.yaml` + README — Render env vars, UptimeRobot `/health` setup
  - Acceptance criteria: render.yaml declares env vars + web service
  - Acceptance criteria: README documents Render deploy + UptimeRobot `/health` ping
  - Scope in: render.yaml blueprint
  - Scope in: README deploy docs
  - Scope out: executing the live deploy
  - Scope out: Dockerfile changes

## Phase 6 — Tests & verification

- [x] Task 19: Add `tests/test_indicators.py` + `test_signals.py` — known series, lookahead safety
  - Acceptance criteria: indicator tests assert known hand-computed series
  - Acceptance criteria: signal tests verify no lookahead (shift(1))
  - Scope in: `test_indicators.py`
  - Scope in: `test_signals.py`
  - Scope out: integration / e2e
  - Scope out: live external API calls

- [x] Task 20: Add `tests/test_filters.py` + `test_sizing.py` — breaker halts at −3%, blackout blocks, sizing hand-calc
  - Acceptance criteria: breaker test halts at day loss ≤ −3%
  - Acceptance criteria: blackout test blocks entries
  - Acceptance criteria: sizing hand-calc matches formula
  - Scope in: `test_filters.py`
  - Scope in: `test_sizing.py`
  - Scope out: live / external API tests
  - Scope out: e2e

- [x] Task 21: Add `tests/test_api.py` + `test_deepseek.py` — `/health` 200, DeepSeek parse failure → `execute=false`
  - Acceptance criteria: `/health` returns 200 (test)
  - Acceptance criteria: DeepSeek parse failure yields `execute=false` (test)
  - Scope in: `test_api.py` (mocked)
  - Scope in: `test_deepseek.py` (mocked)
  - Scope out: live external calls
  - Scope out: endpoints beyond `/health`

- [x] Task 22: Fix `finalize_merged.sh` merge detection — grep `#N[^0-9]` misses issue refs at end of a PR-body line [bug]
- [x] Task 23: Fix `finalize_merged.sh` — `[bug]` tasks can't auto-finalize (`Bug N` rows not mapped to task numbers) [bug]

- [x] Task 24: Update root `strategy.md` + plan docs — record the 10-point strategy review resolutions (lookahead convention, `SQUEEZE_LOOKBACK`, setup lifecycle states, sizing guards)
  - Acceptance criteria: strategy.md documents the lookahead convention (channels/swings shifted 1 bar; indicators read at bar close) and the direction-safe asia TP formula
  - Acceptance criteria: strategy.md adds `SQUEEZE_LOOKBACK` param and lifecycle + sizing-guard sections
  - Acceptance criteria: test_signals.py locks the direction-safe asia TP formula (TP ≤ Asian low on SELL, ≥ Asian high on BUY)
  - Scope in: strategy.md + plan.md + tasks.md documentation
  - Scope in: direction-safe asia TP assertions in `TradingBot/tests/test_signals.py` (locks in §2.3 rule 6)
  - Scope out: any code changes beyond the asia TP test assertions

- [x] Task 25: Implement rolling ATR squeeze lookback in `app/strategy/signals.py` — `BO_SQUEEZE_LOOKBACK=5`, breakout requires any squeeze in the prior N bars
  - Acceptance criteria: `atr_breakout_signals` fires when a squeeze occurred within the prior `BO_SQUEEZE_LOOKBACK` bars (not only the immediate predecessor)
  - Acceptance criteria: no squeeze in the prior N bars blocks the breakout
  - Acceptance criteria: tests added for both cases
  - Scope in: `signals.py` squeeze guard + `BO_SQUEEZE_LOOKBACK` constant + tests
  - Scope out: other strategies / engine

- [~] Task 26: Implement explicit setup lifecycle in `app/core/engine.py` — pending → filled/invalidated/expired with per-strategy invalidation predicates
  - Acceptance criteria: `_pending` becomes a state map; dispatch success marks `filled`
  - Acceptance criteria: a filled/invalidated key frees (strategy, instrument, day, side) for a new setup
  - Acceptance criteria: day rollover marks stale keys `expired`; per-strategy `is_invalidated(df, signal)` predicates added to `signals.py`
  - Acceptance criteria: engine tests cover each transition + re-fire after invalidation
  - Scope in: engine pending lifecycle + invalidation predicates + tests
  - Scope out: order management beyond entry (trailing/cancel/modify)

- [~] Task 27: Implement instrument min/max + margin guards in `app/strategy/sizing.py` — fail-safe None on violation
  - Acceptance criteria: min/max unit map for XAU_USD, EUR_USD, GBP_USD enforced
  - Acceptance criteria: margin check vs `marginAvailable` (fallback notional cap) enforced
  - Acceptance criteria: `build_market_order` returns None on violation; tests added
  - Scope in: sizing guards + config map + tests
  - Scope out: fetching live instrument metadata from OANDA
