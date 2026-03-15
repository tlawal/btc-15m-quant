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
- **Bayesian posterior blending** — logit-space signal weight (0.5), time-decay curve
- **Belief-vol sigma_B** — rolling 3-min belief volatility regresses posterior toward 0.5 in high-noise regimes (capped at 1.15–1.30)
- **EMA score smoothing** — `signed_score = 0.6 * raw + 0.4 * prev` to reduce single-cycle noise
- **Group-max scoring** — 4 signal groups (Trend, Momentum, Flow, Microstructure) each contribute only their strongest member; prevents correlated feature inflation; raw score capped at ±8.0
- **Microstructure signals** — TOB imbalance, CVD velocity, deep OFI (10-level), VPIN proxy
- **Deep LOB + ML ensemble (real-time)** — compute 40-level cumulative imbalance + depth VWAP deviation + momentum-change proxy from Binance L2 (`limit=100`), feed into the lightweight ML model (`inference.py`), and blend with the Bayesian posterior each cycle via weights `ENSEMBLE_BAYES_WEIGHT` and `ENSEMBLE_MODEL_WEIGHT`
- **Hawkes-style timing gate (late window)** — estimate next-event timing from clustered trade arrivals; within last 5 min, if `next_event_sec < 5` and the signal direction agrees, relax required edge down to `HAWKES_LATE_REQUIRED_EDGE`
- **Cross-market perp basis edge override** — compute perp basis `(mark-index)/index` from Binance futures; when `abs(basis_pct) >= BASIS_EDGE_MIN` and basis sign agrees with direction, bypass the `edge_insufficient` gate
- **Tier 1 signals** — Whale flow (Polymarket fills ≥$50), spread skew (NO/YES spread ratio), multi-window momentum
- **Tier 2 signals** — Liquidation cascade (Binance forced orders), funding rate delta
- **Regime-adaptive thresholds** — min score and required edge scale with 15m ATR (low/normal/high vol)
- **Monster signal** — requires BOTH `abs_score >= 8.0 AND posterior >= 0.90`; bypasses early-window guard, uses FOK at ask
- **Late conviction sniping** — within last 3 min, `posterior >= 0.80` and `distance >= $40` suppresses score gate for near-certain OTM binaries
- **Negative-edge block** — never sizes a position when model edge < 0, even on monster signals
- **Early window guard** — blocks non-monster entries for the first 7.5 min of each 15-min window
- **Early-minute momentum (stub)** — concept: BTC price momentum in the first 1-2 minutes of a window predicts direction for the remainder. If BTC moves strongly in one direction immediately after window open, momentum tends to persist. Currently stubbed (`compute_early_minute_momentum()` returns None) pending calibration data from historical windows. Will require: 1m klines from window's first 2 minutes, comparison of close[t=1min] vs open[t=0] as % of ATR, threshold calibration

### Exit Engine (`exit_policy.py`)
Evaluated every 3 seconds. Two tiers: **hard circuit breakers** (no posterior override) and **soft exits** (suppressed by trailing posterior guard while model is still confident).

**Tiered take-profits (percentage from entry):**
- **TP1 (+5%)** — conviction-gated: skipped when `posterior >= TP1_POSTERIOR_CEIL`.
- **TP2 (+15%)** — unconditional: sells **50%** regardless of posterior.
- **TP3 (+20%)** — unconditional: exits **100%** regardless of posterior.

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
3. **Cont, R., Stoikov, S., & Talreja, R. (2014). "A stochastic model for order book dynamics." _Operations Research_, 62(6), 1263–1283.** (order book imbalance and short-horizon price impact)
4. **Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015). "Hawkes processes in finance." _Market Microstructure and Liquidity_, 1(1).** (event-time clustering / intensity models)
5. **Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." _IEEE Transactions on Signal Processing_, 67(11), 3001–3012.** (deep LOB features outperform lagged candle features at short horizons)

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
- `STREAK_HALT_AT = 3` — halts trading after 3 consecutive losses
- `MAX_DAILY_TRADES = 20` — absolute daily trade limit (resets midnight UTC)
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

Approval note (Polymarket conditional tokens):
- Sells require ERC-1155 approval of Polymarket ConditionalTokens to the Exchange contract.
- If you see `conditional_allowance_missing`, you likely need to approve via `setApprovalForAll` and have a small amount of MATIC for gas.
- Polygon mainnet addresses (from `py_clob_client` contract config):
  - ConditionalTokens (ERC-1155): `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
  - Exchange / operator: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`

---

## Manual Redemption of Orphan Shares

Sometimes you may hold resolved conditional tokens in an **EOA wallet** (not the Polymarket proxy wallet). The Polymarket UI cannot claim these because it only sees shares held in the proxy. You can redeem them manually on-chain.

### When this happens
- The market is resolved and you won (e.g., YES paid $1, NO paid $0)
- Your shares are in an EOA (e.g., from a previous script or direct trading)
- The UI shows “no shares” but the EOA still holds the ERC-1155 token

### Option A: Use the provided script (recommended)
Run `redeem_specific_yes.py` with your EOA private key in `config.py`:
```bash
python redeem_specific_yes.py
```
The script calls `redeemPositions` on the CTF Exchange for a specific condition id and waits for the receipt. If you hold a winning share, you’ll receive the USDC payout to the EOA.

### Option B: One‑liner with `cast` (if you have foundry `cast`)
```bash
cast send \
  --rpc-url $POLYGON_RPC_URL \
  --private-key $EOA_PRIVATE_KEY \
  0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de \
  "redeemPositions(bytes32,uint256[])" \
  0xf89e490675e9aa6bf294f26d1bbfb76bcaef4afaf1a3222f216ecc828cdd2247 "[1,2]"
```
Replace the condition id with your market if different.

### Option C: Transfer to proxy wallet (then use UI)
If you prefer the UI, you can `safeTransferFrom` the ERC-1155 from EOA → proxy wallet, then claim via Polymarket. This requires knowing the exact token amount and the proxy address.

### Prerequisites
- **Polygon RPC URL** (Alchemy/QuickNode recommended)
- **EOA private key** (the wallet holding the orphan shares)
- **A tiny bit of MATIC** in the EOA for gas

### Addresses (Polygon mainnet)
- **ConditionalTokens (ERC-1155):** `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- **CTF Exchange (redeemPositions):** `0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de`
- **Exchange/operator (for approvals):** `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`

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

## Profit-Taking

Profit-taking is handled in `exit_policy.py` via tiered take-profits measured as % return from entry price.

- **TP1 (+5%)**: conviction-gated via `TP1_POSTERIOR_CEIL`.
- **TP2 (+15%)**: unconditional partial — sells **50%**.
- **TP3 (+20%)**: unconditional full exit.

This design avoids clipping extremely high-conviction trades too early (TP1 gate), while still guaranteeing you lock in profit at TP2 and fully de-risk at TP3.

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
| `TAKE_PROFIT_PRICE` | 0.99 | Absolute price take-profit (exit when binary >= $0.99) |
| `STOP_LOSS_DELTA` | 7.0 | Score reversal required for ALPHA_DECAY exit |
| `SLIPPAGE_BUFFER_PCT` | 0.008 | Execution haircut on sizing (80bps) |
| `LATE_CONVICTION_POSTERIOR` | 0.80 | Min posterior for late-window sniping |
| `LATE_CONVICTION_DISTANCE` | 40.0 | Min BTC distance from strike for late sniping |
| `BB_SQUEEZE_NORMAL` | 0.0030 | BB squeeze gate threshold (normal regime) |
| `DAILY_LOSS_LIMIT_PCT` | 0.10 | Daily loss limit as fraction of session start balance |

