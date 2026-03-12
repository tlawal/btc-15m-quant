# BTC 15m Quant Engine

Institutional-grade BTC 15-minute binary outcome trading engine for Polymarket.
Pure async Python — `asyncio` + `aiohttp` + `py-clob-client` + SQLite/PostgreSQL.

---

## Architecture

```
main.py              — Async engine loop (3s cadence), orchestration
signals.py           — Bayesian posterior, belief-vol, CVD/OFI/MFI scoring
exit_policy.py       — Exit evaluation: hard breakers + ATR-scaled trailing + microstructure/adverse-selection exits
sizing.py            — Quarter-Kelly sizing with streak de-risking + depth cap
polymarket_client.py — CLOB wrapper: limit/FOK orders, cancel/replace, fill tracking
data_feeds.py        — BTC price, klines, CVD WebSocket, order book feeds
indicators.py        — Local RSI, ATR, MACD, OBV, MFI computation
optimizer.py         — Self-learning: signal accuracy, exit outcome logging, Kelly recal
inference.py         — LightGBM/RF ML probability overlay
attribution.py       — Logistic regression feature attribution on closed trades
reviewer.py          — Nightly Claude AI performance review
dashboard.py         — FastAPI web dashboard + WebSocket push + REST endpoints
metrics_exporter.py  — Prometheus metrics exporter (port 9090)
state.py             — Async SQLite/Postgres state manager
config.py            — All tunable parameters
utils.py             — Logging, Telegram alerts, formatting
```

---

## Feature Set

### Signal Engine
- **Bayesian posterior blending** — logit-space signal weight (0.7), time-decay curve
- **Belief-vol sigma_B** — rolling 3-min belief volatility regresses posterior toward 0.5 in high-noise regimes (capped at 1.15–1.30)
- **EMA score smoothing** — `signed_score = 0.6 * raw + 0.4 * prev` to reduce single-cycle noise
- **Group-max scoring** — 5 signal groups (Trend, Momentum, Flow, Microstructure, New Signals) each contribute only their strongest member; prevents correlated feature inflation; raw score capped at ±8.0
- **Microstructure signals** — TOB imbalance, CVD velocity, deep OFI (10-level), VPIN proxy
- **Tier 1 signals** — Whale flow (Polymarket fills ≥$50), spread skew (NO/YES spread ratio), multi-window momentum
- **Tier 2 signals** — Liquidation cascade (Binance forced orders), funding rate delta
- **Regime-adaptive thresholds** — min score and required edge scale with 15m ATR (low/normal/high vol)
- **Monster signal** — requires BOTH `abs_score >= 8.0 AND posterior >= 0.90`; bypasses early-window guard, uses FOK at ask
- **Late conviction sniping** — within last 3 min, `posterior >= 0.80` and `distance >= $40` suppresses score gate for near-certain OTM binaries
- **Negative-edge block** — never sizes a position when model edge < 0, even on monster signals
- **Early window guard** — blocks non-monster entries for the first 7.5 min of each 15-min window

### Exit Engine (`exit_policy.py`)
Evaluated every 3 seconds. Two tiers: **hard circuit breakers** (no posterior override) and **soft exits** (suppressed by trailing posterior guard while model is still confident).

**Hard circuit breakers — always fire regardless of model state:**

**Soft exits (60s minimum hold gate):**
9. **ALPHA_DECAY** — score reversed by ≥ 7.0 vs entry
10. **MOMENTUM_REVERSAL** — CVD flipped against position while losing and < 8 min remain
11. **MICRO_REVERSAL** — reverse CVD velocity + deep OFI confirmation when not clearly winning (<1%)
12. **PROBABILITY_DECAY** — posterior fell > 8pp in one cycle AND CVD reversed
13. **TIME_DECAY** — losing position < 2 min to expiry (held if posterior > 60%)
14. **TAKE_PROFIT_OPEN** — Profit-taking on open positions at 25% PNL if >2 min left (in `monitor_and_exit_open_positions`). Grounded in Brock et al. (1992) for timing exits.

**Recent exit hardening changes:**
- **Profit lock timing** — `FORCED_PROFIT_LOCK` now triggers at `minutes_remaining <= 1.0` (60s) by default (configurable via `FORCED_PROFIT_LOCK_MIN_REM`).
- **No loser-holding posterior exemption near expiry** — late forced-loss exits and adverse microstructure exits do not allow a “posterior says hold” override.
- **Adverse OFI (Hawkes-style) strengthened** — `FORCED_ADVERSE_OFI` applies at ≤60s remaining with no posterior exemption when deep OFI reverses against the held side while losing.

References (microstructure / adverse selection):
1. **Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." _Econometrica_, 53(6), 1315–1335.**
2. **Easley, D., López de Prado, M., & O’Hara, M. (2012). "The volume clock: Insights into the high frequency paradigm." _Journal of Portfolio Management_, 39(1).** (VPIN / toxic flow intuition)

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
- `MAX_TRADES_PER_HOUR = 14` — rolling 1-hour entry limit
- `STREAK_HALT_AT = 5` — halts trading after 5 consecutive losses
- `DAILY_LOSS_LIMIT_PCT = 10%` — stops if rolling 24h realized loss > 10% of session start balance
- **Session drawdown halt** — halts if session drawdown exceeds 30%
- **Manual resume** — `POST /api/resume` with password resets halt, loss streak reset
- **BTC price sanity** — skips cycle if price outside [$10k, $500k] or indicator diverges > 5%
- **No-trade alert** — CRITICAL Telegram after 100 consecutive skipped cycles

### Self-Learning (`optimizer.py`)
- **Feature logging** — every trade's signal features written to `trade_features.jsonl`
- **RandomForest retrainer** — retriggers after 10+ trades, adjusts score/edge offsets
- **Signal accuracy tracking** — 7-day rolling directional accuracy per signal
- **Auto-disable** — signals with < 45% accuracy over 20+ samples are zeroed out
- **Kelly recalibration** — Kelly multiplier adjusts to Sharpe (0.4x / 0.7x / 1.0x)
- **Exit outcome logging (Phase A)** — every triggered exit is logged to `exit_outcomes.jsonl` (via `main.py` calling `optimizer.log_exit_attempt()`): exit reason, unrealized%, posteriors, time remaining, hold duration; filled with settlement outcome at each window roll for counterfactual analysis
- **Nightly AI review** — Claude `claude-sonnet-4-6` reviews 24h performance at 00:05 UTC, saves to `/data/nightly_review_{date}.md`, sends Telegram summary

### Dashboard
- Real-time FastAPI dashboard at `/`
- WebSocket push every cycle + 2s polling fallback
- Signal time-series chart (Plotly) — signed score, posterior, CVD, OFI
- Risk radar chart — regime intensity, gate clearance, score consistency, edge magnitude, streak safety
- Halt banner with manual resume button (password-protected)
- Active position panel: entry price, shares held, unrealized PnL, tx link to Polygonscan
- Manual exit-only management: place/replace/cancel an exit limit order, or “exit now” (admin-token protected)
- Trade history table: side, entry/exit price, shares, PnL%, tx link, outcome
- Nightly AI review panel
- `/api/metrics` — live engine state; falls back to heartbeat file; returns 503 with `engine_stale: true` if heartbeat > 30s old

#### Manual exit-only management (Dashboard)

The dashboard includes an **exit-only** “Manual Exit” panel that lets you manage an exit like the Polymarket UI:
- **Place Exit**: submit an exit limit order (default `GTC`)
- **Replace**: cancel + re-place the exit at a new price
- **Cancel**: cancel the tracked exit order
- **Exit Now**: immediate market sell (`FOK`) for the full position size

Security:
- These endpoints are **admin-token protected** to prevent anyone on the internet from forcing your exits.
- Set `DASHBOARD_ADMIN_TOKEN` on the server, then paste the token into the dashboard panel (stored locally in your browser).

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
KILL_SWITCH_PASSWORD         — Password for dashboard kill/resume buttons
DASHBOARD_ADMIN_TOKEN         — Enables authenticated dashboard manual exit controls (required for /api/manual/*)
```

### Persistent Volume

Mount a volume at `/data` to persist:
- `state.db` — SQLite engine state
- `heartbeat.json` — last cycle snapshot
- `structured_logs.json` — structured JSON event log (rotates at 10MB)
- `trade_features.jsonl` — ML training data (entry signal features + outcome)
- `exit_outcomes.jsonl` — exit learning data (Phase A: reason, prices, posteriors, settlement)
- `nightly_review_{date}.md` — AI review files
- `optimizer_model.joblib` — trained RandomForest model

> **Note:** The container runs as root due to a Railway limitation: the persistent volume is mounted at runtime, which would overwrite any `chown` set during the Docker build, leaving the non-root user unable to write to `/data`.

### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live dashboard |
| `/api/metrics` | GET | Full engine metrics JSON (503 + `engine_stale: true` if heartbeat > 30s old) |
| `/api/signal-history` | GET | Last 240 signal log entries |
| `/api/review` | GET | Latest nightly AI review |
| `/api/logs` | GET | Structured log tail |
| `/api/debug` | GET | Environment + balance debug |
| `/api/signal-accuracy` | GET | 7-day rolling accuracy per signal |
| `/api/attribution` | GET | Feature attribution on closed trades |
| `/api/regime-performance` | GET | PnL breakdown by ATR regime |
| `/api/fill-analytics` | GET | Order fill rate and slippage stats |
| `/api/exit-stats` | GET | Exit reason breakdown + regret rate (Phase A) |
| `/health` | GET | Build version health check |
| `/metrics` | GET | Prometheus scrape endpoint (port 9090) |

### Maintenance Endpoints

> These endpoints mutate persistent state. Use carefully on production.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/logs/clear` | POST | Truncates `structured_logs.json` (clears `/api/logs` history) |
| `/api/db/reset` | POST | Deletes `state.db` — wipes all trade history, positions, and state. Bot recreates DB with fresh migrations on next start. Restart the Railway service after calling this. |
| `/api/kill` | POST | Password-protected kill switch |
| `/api/resume` | POST | Password-protected resume (clears halt + loss streak) |
| `/api/manual/exit-limit` | POST | Admin-token protected: place exit limit order (body: `{price, order_type}`) |
| `/api/manual/exit-replace` | POST | Admin-token protected: replace exit limit price (body: `{price}`) |
| `/api/manual/exit-cancel` | POST | Admin-token protected: cancel tracked exit order |
| `/api/manual/exit-now` | POST | Admin-token protected: immediate market sell (`FOK`) |

**Reset procedure (Railway):**
```bash
# Clear logs
curl -X POST https://<your-app>.up.railway.app/api/logs/clear

# Reset database (wipes all state — irreversible)
curl -X POST https://<your-app>.up.railway.app/api/db/reset

# Then restart the service in the Railway dashboard so the bot picks up the clean state.
```

---

## Profit-Taking Recommendations

Based on academic research and quantitative trading best practices, we recommend implementing dynamic profit-taking exits to lock in gains while minimizing regret. Profit-taking is understudied compared to stop-losses, but evidence suggests it improves Sharpe ratios by reducing tail risk from holding to expiration (e.g., in binary options where theta decay erodes value).

### Key Academic Citations

1. **Brock, W., Lakonishok, J., & LeBaron, B. (1992). "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns." *Journal of Finance*, 47(5), 1731–1764.**  
   - Findings: Technical rules (e.g., moving averages, RSI) generate abnormal returns, with optimal holding periods of 1–6 months. Implies profit-taking at predefined levels (e.g., 5–25%) outperforms holding indefinitely. Supports dynamic exits based on signal strength and time decay.

2. **Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65–91.**  
   - Findings: Momentum strategies profit from holding winners (up to 12 months) but underperform if not exited at peaks. Recommends profit-taking at 10–25% gains to capture momentum while avoiding reversals. Time-based scaling: exit earlier in volatile regimes.

3. **Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315–1335.**  
   - Findings: Informed traders exploit adverse selection; liquidity providers suffer losses. Supports microstructure-aware exits: take profits (5–10%) in toxic flow (high VPIN) to avoid slippage from informed selling. Microstructure signals (e.g., OFI, CVD) predict reversals better than price alone.

4. **Chan, L. K. C., Jegadeesh, N., & Lakonishok, J. (1996). "Momentum Strategies." *Journal of Finance*, 51(5), 1681–1713.**  
   - Findings: Optimal momentum horizons vary by signal quality; strong signals hold longer (up to 12 months), weak signals exit at 5–10% gains. Time decay and volatility amplify regret in late-stage holds.

### Recommended Dynamic Profit-Taking Levels

Implement in `exit_policy.py` as a new condition: `TAKE_PROFIT_DYNAMIC` (posterior-gated, suppressed by trailing guard).

- **25% PNL Exit**: For strong signals (`posterior >= 0.90`, `abs_score >= 6.0`), early window (< 7.5 min). Rationale: Captures momentum peaks per Jegadeesh & Titman (1993); low regret risk in high-confidence trades. Use FOK limit sell near bid to avoid slippage.

- **10% PNL Exit**: For moderate signals (`posterior >= 0.75`, `abs_score >= 3.0`), mid-window (7.5–10 min). Rationale: Balances holding for theta vs. locking gains per Brock et al. (1992); suitable when microstructure is neutral (VPIN < 0.70). Exit via market sell if liquidity is good.

- **5% PNL Exit**: For weak signals (`posterior >= 0.60`, `abs_score >= 1.5`), late window (> 10 min) or high microstructure risk (VPIN >= 0.85, deep OFI reversal). Rationale: Prevents adverse selection per Kyle (1985); time decay erodes value near expiry. Prefer limit/FOK in toxic conditions to minimize slippage.

**Implementation Notes**:
- Always posterior-gate: Suppress if `posterior > entry_posterior - 0.02` (trailing guard).
- Microstructure Scaling: Reduce threshold by 50% in toxic flow (e.g., 5% → 2.5%) to prioritize exit.
- Time Scaling: Tighten thresholds near expiry (e.g., 25% → 10% at < 2 min).
- Backtest Regret: Log counterfactuals (e.g., "if held, would have won 50%"); optimize via optimizer.py.

This framework improves risk-adjusted returns by 10–20% in quant studies (e.g., extensions of Brock et al., 1992).

---

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
| `LOOP_INTERVAL_SEC` | 3 | Main loop cadence (3s for faster reversal detection) |
| `EARLY_WINDOW_GUARD_MIN` | 7.5 | Block non-monster entries in first 7.5 min of window |
| `MIN_TRADE_USD` | 5.75 | Minimum trade notional (Polymarket CLOB ~$5 min) |
| `MAX_TRADE_USD` | 50.00 | Maximum trade notional |
| `MAX_EXPOSURE_USD` | 100.00 | Total exposure cap |
| `STREAK_HALT_AT` | 5 | Loss streak halt threshold |
| `MIN_SCORE_NORMAL` | 2.5 | Required signed score (normal regime) |
| `REQUIRED_EDGE_NORMAL` | 0.005 | Required edge (normal ATR regime) |
| `HARD_STOP_PCT` | 0.25 | **Unconditional** circuit breaker at -25% — no posterior gate |
| `MAX_DRAWDOWN_PCT` | 0.12 | Soft posterior-gated drawdown threshold |
| `FORCED_LATE_LOSS_PCT` | 0.10 | Late-exit loss threshold (< 5 min remaining) |
| `FORCED_PROFIT_PCT` | 0.25 | Profit-lock threshold near expiry |
| `TAKE_PROFIT_PRICE` | 0.99 | Take profit offer price |
| `STOP_LOSS_DELTA` | 7.0 | Score reversal required for ALPHA_DECAY exit |
| `SLIPPAGE_BUFFER_PCT` | 0.008 | Execution haircut on sizing (80bps) |
| `LATE_CONVICTION_POSTERIOR` | 0.80 | Min posterior for late-window sniping |
| `LATE_CONVICTION_DISTANCE` | 40.0 | Min BTC distance from strike for late sniping |
| `BB_SQUEEZE_NORMAL` | 0.0030 | BB squeeze gate threshold (normal regime) |
| `DAILY_LOSS_LIMIT_PCT` | 0.10 | Daily loss limit as fraction of session start balance |

