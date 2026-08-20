# TradingBot

Dockerized FastAPI trading bot: OANDA prices + Finnhub news → local filters →
DeepSeek JSON veto → 1%-risk MARKET orders on OANDA **practice** accounts.

Full pipeline and configuration reference: repo-root `Plan.md` and
`plans/trading-bot/plan.md`; strategy rules in repo-root `strategy.md`.

## Quickstart

1. `cp .env.example .env` and fill in API keys (practice account first).
2. `docker build -t tradingbot .`
3. `docker run --env-file .env -p 8000:8000 tradingbot`
4. `curl localhost:8000/health` → 200

## Test

```
pytest -q
```
