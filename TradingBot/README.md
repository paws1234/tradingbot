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

## Deploy to Render

The repo ships a Render blueprint ([render.yaml](render.yaml)) that declares the
web service, its env vars, and the `/health` health check. Deploying is a
two-step process: import the blueprint, then fill in the secrets.

### 1. Import the blueprint

1. In the Render dashboard go to **New → Blueprint** and select the
   `paws1234/tradingbot` repository.
2. Render reads `render.yaml`, creates the **tradingbot** web service (Docker
   runtime, free plan), and applies the env defaults from the blueprint.
3. `autoDeploy: true` is set, so every push to the default branch redeploys.
   The Dockerfile builds from `TradingBot/` — no build settings to configure.

### 2. Fill in the secrets

The blueprint declares the following as **secrets** (`sync: false`) — Render
leaves them blank so they are never committed. Set each once in the service's
**Environment** tab:

| Variable | Purpose |
|----------|---------|
| `OANDA_API_KEY` | OANDA v20 API key (practice or live account) |
| `OANDA_ACCOUNT_ID` | OANDA account ID, e.g. `101-004-1234567-001` |
| `DEEPSEEK_API_KEY` | DeepSeek API key for the AI veto gate |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `FINNHUB_API_KEY` | Finnhub API key for news |

All other variables already have sensible defaults from the blueprint
(`ACCOUNT_TYPE=practice`, `STRATEGIES=...`, risk thresholds, etc.) and can be
overridden per-environment if needed. Keep `ACCOUNT_TYPE=practice` unless you
explicitly opt into live trading.

### 3. Keep it awake with UptimeRobot

Render free web services sleep after ~15 minutes without inbound traffic and
wake on the next request (slow first hit). UptimeRobot keeps the service warm
and verifies it is healthy:

1. Create a free UptimeRobot account and add a **New monitor**:
   - Monitor type: **HTTP(s)**
   - URL: `https://<service-name>.onrender.com/health`
   - Interval: **5 minutes** (or your preferred cadence)
2. Set **Monitoring Alert Contacts** to whatever you use (email, Slack, etc.)
   so a `200` regression or downtime notifies you.
3. UptimeRobot's GET request hits the FastAPI `/health` route every interval,
   keeping the service from sleeping and confirming it responds.

`/health` returns `200` with a JSON status body while the app is up; Render's
own health checks use the same path, so the monitor and the platform agree on
what "healthy" means.

> Local quickstart (above) does not require any of this — Render is only the
> hosted deployment path.
