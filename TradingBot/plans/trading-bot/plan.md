approved: true
# Plan: trading-bot — Dockerized OANDA + DeepSeek Trading Bot

> Requirements source: root `Plan.md` (pipeline, phases, env vars) + root
> `strategy.md` (four strategies, registry, signal contract).

## Goal

Build a fully Dockerized FastAPI trading bot that streams OANDA prices and
Finnhub news, filters setups through local rules and a mandatory DeepSeek JSON
veto gate, and dispatches 1%-risk MARKET orders with SL/TP on OANDA practice
accounts.

## Scope

In:

- FastAPI service (`/health`, `/status`), kept alive by UptimeRobot pings
- OANDA v20: candles/balance/orders REST + pricing stream (HTTP chunked NDJSON, auto-reconnect)
- Finnhub news (WS + REST polling fallback); ForexFactory calendar scraper with manual override
- Four M15 strategies from `strategy.md` (asia_sweep, ema_fvg, atr_breakout,
  mean_reversion) via a registry; closed bars only, `shift(1)` lookahead safety,
  dedup per (strategy, instrument, day, side)
- Stage 2 local filters: circuit breaker (day loss ≤ -3%), news blackout (±30 min)
- DeepSeek JSON veto gate, fail-safe `execute=false` on parse failure, audit-logged
- Sizing: `units = (balance × 0.01) / |entry − SL|`, MARKET orders with
  `stopLossOnFill` / `takeProfitOnFill`
- MongoDB Atlas state: `daily_context`, `account_state`, `trade_logs`, `signals`
- Docker packaging (`python:3.12-slim`, non-root, HEALTHCHECK), `render.yaml` blueprint
- pytest suite with mocked external APIs (respx)

Out:

- Backtesting / historical optimization
- Order lifecycle beyond entry (no trailing/cancel/modify loop in v1)
- Live orders unless `ACCOUNT_TYPE=live` is explicitly set
- Web UI / dashboard

## Architecture

Render web service (Docker) running FastAPI + asyncio background engine.

Pipeline: daily context (00:00 UTC) → streams (OANDA M15, Finnhub) → strategy
signals (pandas, M15 closed bars, H1 trend filter) → circuit breaker → news
blackout → DeepSeek veto → 1% risk MARKET order with SL/TP on fill.

State in MongoDB Atlas via motor; config via pydantic-settings (fail-fast).

Locked decisions:

- `python:3.12-slim`, non-root, pinned `requirements.txt`
- OANDA streaming: `aiter_lines()` NDJSON, `PRICE` vs `HEARTBEAT`, backoff reconnect
- DeepSeek: openai SDK, `base_url` `https://api.deepseek.com`, model `deepseek-chat`
- `STRATEGIES` env selects active strategies (default all four);
  instrument→strategy mapping defaults per `strategy.md` §6.3
- Every signal carries a numeric `stop_loss` and `pending_ai_veto=True`
- App code lives in `TradingBot/` inside the repo root
- Strategy review resolutions recorded in root `strategy.md` §1.6/§2.3/§4.2/
  §6.4/§6.5 (Task 24 docs); Tasks 25–27 implement the code changes

## Requirements

- Pipeline, env vars, Mongo collections, phases per root `Plan.md`
- Four strategies with exact params, registry, dedup, lookahead-safety,
  mandatory AI veto per `strategy.md`
- Practice accounts by default
- Verification: `docker build`/`run` + `/health` 200; pytest green on known
  series, breaker halt at −3%, blackout block, sizing hand-calc, DeepSeek fail-safe

## Phases

- Phase 0 — Scaffolding & configuration
- Phase 1 — Data layer
- Phase 2 — Indicators & strategies
- Phase 3 — DeepSeek gate
- Phase 4 — Engine & FastAPI
- Phase 5 — Docker & deploy files
- Phase 6 — Tests & verification
