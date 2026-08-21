# Issue mapping — trading-bot

Host: GitHub | Repo: paws1234/tradingbot | Default branch: main

> This mapping is created at `/plan` time (one issue per task) and updated by
> `/implement`, `/review`, `/test`, and `/amend` (see CLAUDE.md §Git Host
> Integration). `[bug]` rows are out-of-scope issues captured automatically.
> Status reflects the task state in `tasks.md` and the issue state on the host.

| Task | Issue | Status | Title |
|------|-------|--------|-------|
| Task 1 | #5 | closed | Scaffold the `TradingBot/` project — app package, `tests/`, `requirements.txt`, `.env.example`, `.dockerignore`, README stub |
| Task 2 | #6 | closed | Implement `app/config.py` — pydantic-settings env validation (OANDA practice/live switch, INSTRUMENTS + STRATEGIES parsing, risk thresholds, DeepSeek defaults) |
| Task 3 | #7 | closed | Define `app/models/schemas.py` — Candle, PriceTick, NewsItem, CalendarEvent, Signal, TradeDecision, OrderResult pydantic models |
| Task 4 | #8 | closed | Implement `app/data/mongo.py` — motor client; collections `daily_context`, `account_state`, `trade_logs`, `signals`; upsert helpers |
| Task 5 | #9 | closed | Implement `app/data/oanda.py` — REST (candles, balance, orders) + NDJSON pricing stream with auto-reconnect |
| Task 6 | #10 | closed | Implement `app/data/finnhub.py` — news WebSocket with REST polling fallback |
| Task 7 | #11 | closed | Implement `app/data/forexfactory.py` — calendar scraper (httpx + bs4, EST→UTC, retries, parse tolerance, manual override) |
| Task 8 | #12 | closed | Implement `app/indicators/technical.py` — ema, sma, rsi, true_range, atr, bollinger, adx, donchian, fvg, session_range, m15→h1 resample |
| Task 9 | #13 | closed | Implement `app/strategy/signals.py` — four strategy signal functions + STRATEGY_REGISTRY + dedup |
| Task 10 | #14 | closed | Implement `app/strategy/filters.py` — circuit breaker halting at day loss ≤ −3%, and ±30 min news blackout |
| Task 11 | #15 | closed | Implement `app/strategy/sizing.py` — 1% risk sizing, OANDA unit rounding, SL/TP, MARKET payload |
| Task 12 | #16 | closed | Implement `app/data/deepseek.py` — openai SDK wrapper, JSON response_format, retry/backoff, parse fail-safe `execute=false` |
| Task 13 | #17 | closed | Audit-log DeepSeek decisions + order results to `trade_logs` |
| Task 14 | #18 | closed | Implement `app/core/scheduler.py` — AsyncIOScheduler daily 00:00 UTC context build + `account_state` reset |
| Task 15 | #19 | closed | Implement `app/core/engine.py` — per-instrument stream tasks, candle builder, signals → filters → DeepSeek → dispatch |
| Task 16 | #20 | closed | Implement `app/api/routes.py` + `app/main.py` — `/health`, `/status`, lifespan start/stop with graceful cancel |
| Task 17 | #21 | closed | Add `Dockerfile` — python:3.12-slim, non-root, EXPOSE, HEALTHCHECK, uvicorn CMD |
| Task 18 | #22 | closed | Add `render.yaml` + README — Render env vars, UptimeRobot `/health` setup |
| Task 19 | #23 | closed | Add `tests/test_indicators.py` + `test_signals.py` — known series, lookahead safety |
| Task 20 | #24 | closed | Add `tests/test_filters.py` + `test_sizing.py` — breaker halts at −3%, blackout blocks, sizing hand-calc |
| Task 21 | #25 | closed | Add `tests/test_api.py` + `test_deepseek.py` — `/health` 200, DeepSeek parse failure → `execute=false` |
| Bug 1 | #29 | closed | Fix `finalize_merged.sh` merge detection — grep `#N[^0-9]` misses issue refs at end of a PR-body line |
| Bug 2 | #32 | closed | Fix `finalize_merged.sh` — `[bug]` tasks can't auto-finalize (`Bug N` rows not mapped to task numbers) |
| Task 24 | #34 | closed | Update root `strategy.md` + plan docs — record the 10-point strategy review resolutions (lookahead convention, `SQUEEZE_LOOKBACK`, setup lifecycle states, sizing guards) |
| Task 25 | #49 | open | Implement rolling ATR squeeze lookback in `app/strategy/signals.py` — `BO_SQUEEZE_LOOKBACK=5`, breakout requires any squeeze in the prior N bars |
