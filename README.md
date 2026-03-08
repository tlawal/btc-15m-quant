# BTC 15m Quant Engine

Institutional-grade BTC 15-minute binary outcome trading engine for Polymarket.
Pure async Python — `asyncio` + `aiohttp` + `py-clob-client` + SQLite/PostgreSQL.

---

## Architecture

```
main.py          — Async engine loop (15s cadence), orchestration
signals.py       — Bayesian posterior, belief-vol, CVD/OFI/MFI scoring
exit_policy.py   — Exit evaluation: drawdown, alpha decay, momentum reversal
sizing.py        — Quarter-Kelly sizing with streak de-risking + depth cap
polymarket_client.py — CLOB wrapper: limit/FOK orders, cancel/replace, fill tracking
data_feeds.py    — BTC price, klines, CVD WebSocket, order book feeds
indicators.py    — Local RSI, ATR, MACD, OBV, MFI computation
optimizer.py     — Self-learning: signal accuracy tracking, auto-disable, Kelly recal
inference.py     — LightGBM/RF ML probability overlay
attribution.py   — Logistic regression feature attribution on closed trades
reviewer.py      — Nightly Claude AI performance review
dashboard.py     — FastAPI web dashboard + WebSocket push + REST endpoints
metrics_exporter.py — Prometheus metrics exporter (port 9090)
state.py         — Async SQLite/Postgres state manager
config.py        — All tunable parameters
utils.py         — Logging, Telegram alerts, formatting
```

---

## Feature Set

### Signal Engine
- **Bayesian posterior blending** — logit-space signal weight (0.7), time-decay curve
- **Belief-vol sigma_B** — rolling 3-min belief volatility regresses posterior toward 0.5 in high-noise regimes (capped at 1.0)
- **EMA score smoothing** — `signed_score = 0.6 * raw + 0.4 * prev` to reduce single-cycle noise
- **Group-max scoring** — 5 signal groups (Trend, Momentum, Flow, Microstructure, New Signals) each contribute only their strongest member; prevents correlated feature inflation; raw score capped at ±8.0
- **Microstructure signals** — TOB imbalance, CVD velocity, deep OFI (10-level), VPIN proxy
- **Tier 1 signals** — Whale flow (Polymarket fills ≥$50), spread skew (NO/YES spread ratio), multi-window momentum
- **Tier 2 signals** — Liquidation cascade (Binance forced orders), funding rate delta
- **Regime-adaptive thresholds** — min score and required edge scale with 15m ATR (low/normal/high)
- **Monster signal** — requires BOTH `abs_score >= 8.0 AND posterior >= 0.90`; uses FOK at ask
- **Negative-edge block** — never sizes a position when model edge < 0, even on monster signals

### Execution
- **Smart entry pricing** — passive `bid+1tick` for GTC, aggressive `ask` for FOK monster signals
- **Mid-price fallback** — when ask is None (deep ITM), uses `mid + 0.01` capped at 0.99
- **Depth-aware sizing** — caps position at 50% of top-of-book depth
- **Stale order replace** — cancel + re-place after 12s if entry GTC not filled
- **Exit timeout FOK** — if exit order pending > 60s, force-replaces as FOK at `bid - 1 tick`
- **State checkpoint** — saves state before every order placement (crash-safe)
- **Startup reconciliation** — checks pending orders AND live API positions on restart; populates `held_position` from Polymarket if local state is out of sync
- **Slippage tracking** — actual fill price vs intended price logged per trade; warns if > 1%
- **Market quality filter** — skips cycle if spreads > 8%, book depth < 20 USDC, or klines > 5 min stale

### Risk Management
- `MIN_TRADE_USD = 5.75` — minimum notional per trade (Polymarket CLOB minimum ~$5)
- `MAX_TRADE_USD = 50.00` — absolute per-trade cap
- `MAX_EXPOSURE_USD = 100.00` — blocks new entries if already at exposure limit
- `MAX_TRADES_PER_HOUR = 8` — rolling 1-hour entry limit
- `STREAK_HALT_AT = 3` — halts trading after 3 consecutive losses
- `DAILY_LOSS_LIMIT_PCT = 10%` — stops if rolling 24h realized loss > 10% of session start balance
- **BTC price sanity** — skips cycle if price outside [$10k, $500k] or indicator diverges > 5%
- **No-trade alert** — CRITICAL Telegram after 100 consecutive skipped cycles

### Self-Learning
- **Feature logging** — every trade's signal features written to `trade_features.jsonl`
- **RandomForest retrainer** — retriggers after 10+ trades, adjusts score/edge offsets
- **Signal accuracy tracking** — 7-day rolling directional accuracy per signal
- **Auto-disable** — signals with < 45% accuracy over 20+ samples are zeroed out
- **Kelly recalibration** — Kelly multiplier adjusts to Sharpe (0.4x / 0.7x / 1.0x)
- **Nightly AI review** — Claude `claude-sonnet-4-6` reviews 24h performance at 00:05 UTC, saves to `/data/nightly_review_{date}.md`, sends Telegram summary

### Dashboard
- Real-time FastAPI dashboard at `/`
- WebSocket push every cycle + 2s polling fallback
- Signal time-series chart (Plotly) — signed score, posterior, CVD, OFI
- Risk radar chart — regime intensity, gate clearance, score consistency, edge magnitude, streak safety
- Nightly AI review panel
- `/api/metrics` — live engine state (never 503, falls back to heartbeat file)
- `/api/signal-history` — last 240 structured log entries for chart rendering
- `/api/review` — latest nightly markdown review
- `/api/logs` — last N structured log entries
- `/api/debug` — full environment + balance debug
- `/api/kill` — password-protected kill switch

---

## Deployment (Railway)

### Required Environment Variables

```
POLYMARKET_PRIVATE_KEY       — Polygon wallet private key (0x...)
POLYMARKET_API_KEY           — Polymarket CLOB API key
POLYMARKET_API_SECRET        — Polymarket CLOB API secret
POLYMARKET_API_PASSPHRASE    — Polymarket CLOB passphrase
POLYGON_RPC_URL              — Polygon mainnet RPC (Alchemy/QuickNode recommended)
TELEGRAM_TOKEN               — Bot token from @BotFather
TELEGRAM_CHAT_ID             — Target chat/channel ID
```

### Optional

```
ANTHROPIC_API_KEY            — Enables nightly Claude AI review
BINANCE_API_KEY              — Binance API (public endpoints work without)
COINBASE_API_KEY             — Coinbase candle fallback
DATABASE_URL                 — Default: sqlite+aiosqlite:////data/state.db
LOG_LEVEL                    — Default: INFO
PAPER_TRADE_ENABLED          — Set "true" for paper trading (no real orders placed)
KILL_SWITCH                  — Set "true" to halt all entries immediately
KILL_SWITCH_PASSWORD         — Password for dashboard kill switch button
```

### Persistent Volume

Mount a volume at `/data` to persist:
- `state.db` — SQLite engine state
- `heartbeat.json` — last cycle snapshot
- `structured_logs.json` — structured JSON event log (rotates at 10MB)
- `trade_features.jsonl` — ML training data
- `nightly_review_{date}.md` — AI review files
- `optimizer_model.joblib` — trained RandomForest model

> **Note:** The container runs as root due to a Railway limitation: the persistent volume is mounted at runtime, which would overwrite any `chown` set during the Docker build, leaving the non-root user unable to write to `/data`.

### Monitoring Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Live dashboard |
| `/api/metrics` | Full engine metrics JSON |
| `/api/signal-history` | Last 240 signal log entries |
| `/api/review` | Latest nightly AI review |
| `/api/logs` | Structured log tail |
| `/api/debug` | Environment + balance debug |
| `/health` | Build version health check |
| `/metrics` | Prometheus scrape endpoint (port 9090) |

---

## Local Development

```bash
cp .env.example .env
# Fill in credentials
pip install -r requirements.txt
python main.py

# Paper trade (no real orders):
PAPER_TRADE_ENABLED=true python main.py

# Reset state:
python main.py --reset
```

---

## Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_TRADE_USD` | 5.75 | Minimum trade notional (Polymarket CLOB ~$5 min) |
| `MAX_TRADE_USD` | 50.00 | Maximum trade notional |
| `MAX_EXPOSURE_USD` | 100.00 | Total exposure cap |
| `STREAK_HALT_AT` | 3 | Loss streak halt threshold |
| `MIN_SCORE_NORMAL` | 2.5 | Required signed score (normal regime) |
| `REQUIRED_EDGE_NORMAL` | 0.035 | Required edge (normal regime) |
| `SLIPPAGE_BUFFER_PCT` | 0.008 | Execution haircut on sizing |
| `LOOP_INTERVAL_SEC` | 15 | Main loop cadence |
| `BB_SQUEEZE_LOW` | 0.0015 | BB squeeze threshold (low-vol regime) |
| `BB_SQUEEZE_NORMAL` | 0.0030 | BB squeeze threshold (normal regime) |
| `BB_SQUEEZE_HIGH` | 0.0050 | BB squeeze threshold (high-vol regime) |
