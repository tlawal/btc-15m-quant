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
- **Bayesian posterior blending** — logit-space signal weight (0.30 after Fix #4), time-decay curve
- **Student-T CDF (df=6)** — fat-tail corrected probability from z-score (Fix #3 replacing normal CDF)
- **Belief-vol sigma_B** — rolling 3-min belief volatility regresses posterior toward 0.5 in high-noise regimes (capped at 1.15–1.30)
- **Noise Debouncing & Hysteresis** — Requires the ML model to hold a confirmed directional score for 3+ consecutive cycles (9 to 15 seconds) before entry; instantly rejects transient fakeouts.
- **EMA score smoothing** — `signed_score = 0.6 * raw + 0.4 * prev` to reduce single-cycle noise
- **Group-max scoring** — 4 signal groups (Trend, Momentum, Flow, Microstructure) each contribute only their strongest member; prevents correlated feature inflation; raw score capped at ±8.0
- **Microstructure signals** — TOB imbalance, CVD velocity, deep OFI (10-level), VPIN proxy
- **Deep LOB + ML ensemble (real-time)** — compute 40-level cumulative imbalance + depth VWAP deviation + momentum-change proxy from Binance L2 (`limit=100`), feed into the lightweight ML model (`inference.py`), and blend with the Bayesian posterior each cycle via weights `ENSEMBLE_BAYES_WEIGHT` and `ENSEMBLE_MODEL_WEIGHT`
- **Hawkes-style timing gate (late window)** — estimate next-event timing from clustered trade arrivals; within last 5 min, if `next_event_sec < 5` and the signal direction agrees, relax required edge down to `HAWKES_LATE_REQUIRED_EDGE`
- **Cross-market perp basis edge override** — compute perp basis `(mark-index)/index` from Binance futures; when `abs(basis_pct) >= BASIS_EDGE_MIN` and basis sign agrees with direction, bypass the `edge_insufficient` gate
- **Tier 1 signals** — Whale flow (Polymarket fills ≥$50), spread skew (NO/YES spread ratio), multi-window momentum
- **Tier 2 signals** — Liquidation cascade (Binance forced orders), funding rate delta
- **Regime-adaptive thresholds** — min score and required edge scale with 15m ATR (low/normal/high vol)
- **BTC momentum velocity gate** — blocks entry when BTC moves > `MOMENTUM_GATE_ATR_THRESHOLD` (0.25) ATR in 15s against the trade direction, using `state.prev_price3` (3-cycle lookback). Prevents entering into an adverse momentum surge where posterior hasn't caught up yet. Ref: Almgren & Chriss (2001).
- **Polymarket LOB adverse imbalance gate** — blocks entry when `ask_size / (bid_size + ask_size) > PM_LOB_ADVERSE_THRESHOLD` (0.80) on the PM book; heavy sell-side skew signals informed distribution. Ref: Cont, Kukanov & Stoikov (2014).
- **Funding rate adverse gate** — blocks entry when the Binance perpetual funding rate opposes the trade direction by more than `FUNDING_RATE_GATE_THRESHOLD` (0.0002 = 0.02%/8h); positive funding = bulls in control, adverse for DOWN trades. Ref: Liu & Tsyvinski (2021).
- **Monster signal** — requires BOTH `abs_score >= 8.0 AND posterior >= 0.90`; bypasses early-window guard, uses FOK at ask
- **Late conviction sniping** — within last 3 min, `posterior >= 0.80` and `distance >= $40` suppresses score gate for near-certain OTM binaries
- **Negative-edge block** — never sizes a position when model edge < 0, even on monster signals
- **Early window guard** — blocks non-monster entries for the first 7.5 min of each 15-min window
- **Early-minute momentum (stub)** — concept: BTC price momentum in the first 1-2 minutes of a window predicts direction for the remainder. If BTC moves strongly in one direction immediately after window open, momentum tends to persist. Currently stubbed (`compute_early_minute_momentum()` returns None) pending calibration data from historical windows. Will require: 1m klines from window's first 2 minutes, comparison of close[t=1min] vs open[t=0] as % of ATR, threshold calibration

### Audit-Driven Signal Fixes (Forensic Audit — March 2026)

All 10 findings from the institutional forensic audit are implemented:

| Fix | Change | File |
|-----|--------|------|
| **#1** | `STRIKE_DISTANCE_EXCEEDED` grace period: 60s → **20s** | `exit_policy.py` |
| **#2** | `MIN_ENTRY_DISTANCE_ATR_MULT`: 0.25 → **0.40** | `config.py` |
| **#3** | Replace `normal_cdf` with **Student-T CDF (df=6)** | `signals.py` |
| **#4** | Bayesian market weight: 0.50 → **0.30** (`BAYES_SIGNAL_WEIGHT`) | `signals.py` |
| **#5** | **Isotonic calibration** scaffold (`calibration.py`) | `calibration.py` |
| **#6** | **1H trend gate** — blocks entries opposing EMA9/EMA20 crossover on hourly klines | `indicators.py`, `main.py`, `signals.py` |
| **#7** | **Reverse convergence exit** — exits when opposing side bid ≥ 0.85 | `exit_policy.py`, `main.py` |
| **#8** | **Emergency market-sell** — FOK at market if loss > 8% and first sell failed | `main.py` |
| **#9** | **VPOC proximity gate** — blocks entry when BTC is >1.5% from volume POC | `data_feeds.py`, `signals.py` |
| **#10** | **Candlestick pattern gate** — blocks entry on engulfing/shooting star/hammer reversal signals | `indicators.py`, `signals.py` |

### Exit Engine (`exit_policy.py`)
Evaluated every 3 seconds. Two tiers: **hard circuit breakers** (no posterior override) and **soft exits** (suppressed by trailing posterior guard while model is still confident).

**Tiered take-profits (percentage from entry):**
- **TP1 (+5%)** — conviction-gated and **adaptive sizing**: skipped when `posterior >= TP1_POSTERIOR_CEIL`; otherwise sells between **33% and 100%** based on strike proximity, time remaining, and signal degradation vs entry (see `TP1_*` knobs in `config.py`).
- **TP2 (+15%)** — unconditional: sells **1/3** (when `TP_PARTIAL_ENABLED=true`).
- **TP3 (+20%)** — unconditional: exits **100%** regardless of posterior.

**Hard circuit breakers — always fire regardless of model state:**

- **TRAIL_PRICE_STOP** — Dynamic trailing stop triggered when trade reaches +5% profit, unconditionally locking in gains if price sheds 10% from the highest recorded peak (MFE tracker). Protects winning trades from falling back down to the -25% hard stop.
- **TP1_TRAIL_PRICE_STOP** — After **TP1 fill**, enables a **tighter** price-based trailing stop on the remainder (based on the same MFE tracker). This reduces the probability that a partial profit quickly reverses into a large loss on the runner.
- **STRIKE_DISTANCE_EXCEEDED** — fires when `|btc_price - strike| > multiplier × ATR14` while losing (unrealized < -5%). Multiplier is ATR-adaptive: **0.40×** in high-vol regime (ATR > 200), **0.60×** normal, **0.80×** low-vol. Tighter threshold in fast-moving regimes; wider room in quiet ones. Ref: Avellaneda & Stoikov (2008).

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

References:

**Entry protection / adverse selection:**
0. **Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." _Journal of Risk_, 3(2), 5–39.** (adverse selection cost spikes when underlying momentum opposes position direction — basis for momentum velocity gate)
0. **Liu, Y. & Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency." _Review of Financial Studies_, 34(6), 2689–2727.** (BTC perpetual funding rate has 24h directional predictive power — basis for funding rate gate)

**Microstructure / adverse selection:**
1. **Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." _Econometrica_, 53(6), 1315–1335.**
2. **Easley, D., López de Prado, M., & O’Hara, M. (2012). "The volume clock: Insights into the high frequency paradigm." _Journal of Portfolio Management_, 39(1).** (VPIN / toxic flow intuition)
3. **Cont, R., Stoikov, S., & Talreja, R. (2014). "A stochastic model for order book dynamics." _Operations Research_, 62(6), 1263–1283.** (order book imbalance and short-horizon price impact)
4. **Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015). "Hawkes processes in finance." _Market Microstructure and Liquidity_, 1(1).** (event-time clustering / intensity models)
5. **Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." _IEEE Transactions on Signal Processing_, 67(11), 3001–3012.** (deep LOB features outperform lagged candle features)

**Prediction markets:**
6. **Pennock, D. M. & Sami, R. (2007). "Computational Aspects of Prediction Markets." Ch. 26 in _Algorithmic Game Theory_, Cambridge University Press.** (transient spikes in thin prediction markets are noise; multi-observation confirmation)
7. **Hanson, R. (2003). "Combinatorial Information Market Design." _Information Systems Frontiers_, 5(1), 107–119.** (logarithmic market scoring rules; single-tick jumps unreliable in thin markets)
8. **Tetlock, P. E. (2004). "How Efficient Are Information Markets? Evidence from Prediction Markets." Mimeo.** (prediction market prices overshoot then mean-revert)
9. **Das, S. & Magdon-Ismail, M. (2009). "Adapting to a Market Shock: Optimal Sequential Market-Making." _NIPS 2009_.** (adverse selection from informed traders near expiry)

**Optimal order placement:**
10. **Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book." _Quantitative Finance_, 8(3), 217–224.** (optimal bid/ask placement; reservation price offset proportional to volatility × time)
11. **Cont, R. & Kukanov, A. (2017). "Optimal order placement in limit order markets." _Quantitative Finance_, 17(8), 1143–1156.** (aggressive fills when alpha strong + time short; passive when moderate signal + time permits)
12. **Gueant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). "Dealing with the Inventory Risk." _Mathematics and Financial Economics_, 7(4), 477–507.** (optimal offset from fair value larger in thin markets)

**Execution near discrete-time events:**
13. **Budish, E., Cramton, P. & Shim, J. (2015). "The High-Frequency Trading Arms Race." _Quarterly Journal of Economics_, 130(4), 1547–1621.** (sniping/latency arbitrage intensifies near discrete-time market events)
14. **Gjerstad, S. & Dickhaut, J. (1998). "Price Formation in Double Auctions." _Games and Economic Behavior_, 22(1), 1–29.** (order book transience in thin markets; displayed price ≠ executable price)

### Late-Window Entry Hardening
Addresses adverse selection and stale-fill vulnerabilities in the final minutes of each 15-minute window.

- **FOK for late-window entries** — entries with < 4 min remaining use Fill-or-Kill instead of GTC. GTC orders in thin late-window books hang 10-15s and fill at stale prices after the market has already moved. Per Budish, Cramton & Shim (2015), sniping and latency arbitrage intensify near discrete-time events.
- **One-sided gate multi-cycle confirmation** — the `YES >= 0.75` / `NO >= 0.75` gate now requires 2+ consecutive cycles of clearance before allowing entry. A single transient pump (e.g. 0.72→0.82 in one 3s cycle) no longer clears the gate. Per Pennock & Sami (2007) and Hanson (2003), single-tick price jumps in thin prediction markets are noise — sustained movement is the signal.
- **FORCED_DRAWDOWN grace period** — within the first 15s after fill, FORCED_DRAWDOWN only fires if the loss exceeds the HARD_STOP threshold (-25%) or model conviction dropped >10pp. Hard stops (Layer 0/1) are unaffected. Per Tetlock (2004), prediction market prices overshoot then mean-revert — a brief grace period allows the initial shock to resolve.
- **MAE tightening grace exemption** — the MAE ×0.60 tightening factor only applies after the grace period. Instant post-fill drawdowns represent price discovery, not a "second adverse move."

### Adaptive Entry Pricing
- **Pump reversion detection** — when the relevant side's mid-price pumps >5% in a single cycle, the bot places a limit buy $0.03 below mid instead of at `bid + $0.01`. Captures mean-reversion instead of buying at the peak. Per Tetlock (2004) and Avellaneda & Stoikov (2008).
- **FOK vs passive limit decision framework** — monster signals or late-window (<4 min) → FOK at ask; pump detected → limit below mid; normal → passive `bid + tick`. Based on Cont & Kukanov (2017): aggressive fills when alpha is strong + time is short; passive when signal is moderate + time permits.
- **Data-driven offset optimization (planned)** — optimizer.py will log fill rates and entry conditions, and tune `PUMP_REVERSION_OFFSET` adaptively based on historical fill probability vs. edge captured. Per Gueant, Lehalle & Fernandez-Tapia (2013), the optimal offset from fair value is larger in thin markets (low order arrival intensity).

### Execution
- **Smart entry pricing** — passive `bid+1tick` for GTC, aggressive `ask` for FOK monster signals, reversion `mid - offset` on pump detection
- **Mid-price fallback** — when ask is None (deep ITM), uses `mid + 0.01` capped at 0.99
- **Depth-aware sizing** — caps position at 50% of top-of-book depth
- **Stale order replace** — cancel + re-place after 12s if entry GTC not filled
- **Exit timeout FOK** — if exit order pending > 60s, force-replaces as FOK at `bid - 1 tick`
- **FOK immediate reconcile recheck** — for FOK exits, the engine re-checks order status after a short delay (`FOK_RECHECK_DELAY_SEC`) before concluding the order was killed (eventual-consistency hardening).
- **State checkpoint** — saves state before every order placement (crash-safe)
- **Startup reconciliation** — checks pending orders AND live API positions on restart; populates `held_position` from Polymarket if local state is out of sync
- **On-chain position reconciliation** — on startup and before every sell, queries on-chain ERC-1155 `balanceOf` (Gnosis CTF contract) to verify inherited position size is correct. Clears phantom positions (on-chain balance = 0) immediately. Catches stale Polymarket positions-API data that can cause futile sell loops.
- **Sell retry cap** — limits sell attempts to `MAX_CONSECUTIVE_SELL_FAILURES` (default 20) total; after that, stops retrying and waits for auto-settle. Prevents 3+ minute loops of futile HTTP calls when shares don't exist on-chain.
- **Sell failure diagnostics** — on sell failure, logs `SELL_FAIL_DEBUG` with positions-API response and on-chain balance side-by-side for root cause analysis.
- **Full-position sell rounding** — `limit_sell` uses `round(size)` (nearest integer) instead of `int(size)` (floor) for sizes > 1 share. `int(6.9905)` silently dropped 0.9905 shares per partial exit, leaving a rump that then triggered a separate HARD_STOP exit at a worse price.
- **Dust write-off guard** — if `sell_size < MIN_SELL_SIZE (0.05 shares)` after all on-chain/API clamping, the position is written off as `DUST_WRITEOFF` with blended PnL rather than placing an order that will fail with a 400 allowance error.
- **Fractional Dust Eradication** — `pos.size` subtractions on partial fills are strictly subjected to `math.floor(size * 10000) / 10000`, matching Polymarket's token precision limit natively and completely neutralizing recursive `DUST_SKIP` 82,000% PNL UI math bugs.
- **Blended PnL on auto-settle** — `AUTO_SETTLE_WIN` and `AUTO_SETTLE_LOSS` now compute `blended_exit_price` and `blended_pnl` weighted across all partial exits plus the remaining shares at settlement ($1.00 or $0.00). Previously, auto-settle overwrote `pnl = -1.0` even when 90%+ of the position had already been sold at recovery prices, causing the dashboard to show -100% instead of the real ~-15%.
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
- `/api/tx/{tx_hash}` — debug/forensics: fetch Polygon transaction receipt JSON for on-chain ground truth when reconciling missed fills.

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

- **TP1 (+5%)**: conviction-gated via `TP1_POSTERIOR_CEIL`, and **adaptive** partial exit sizing:
  - base/mid/max fractions: `TP1_PARTIAL_BASE`, `TP1_PARTIAL_MID`, `TP1_PARTIAL_MAX` (max may be **1.0** = full exit)
  - increases TP1 sell fraction when:
    - near strike: `abs(distance) <= TP1_CLOSE_DIST`
    - late in window: `minutes_remaining <= TP1_LATE_MIN_REM`
    - signal degraded vs entry: posterior drop `>= TP1_POST_DROP_THRESH` and/or edge drop `>= TP1_EDGE_DROP_THRESH`
- **TP2 (+15%)**: unconditional partial — sells **1/3** (when `TP_PARTIAL_ENABLED=true`).
- **TP3 (+20%)**: unconditional full exit.

After TP1 fill, a tighter trailing stop can protect the runner:
- `TP1_TRAIL_PRICE_ACTIVATION_PCT` arms the tighter trailing once MFE exceeds this profit.
- `TP1_TRAIL_PRICE_DISTANCE_PCT` sets how far below peak price to trail.

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
| `TP1_PARTIAL_BASE` | 0.333 | Adaptive TP1: base sell fraction when conditions are favorable |
| `TP1_PARTIAL_MID` | 0.666 | Adaptive TP1: sell fraction when near strike / late / degraded |
| `TP1_PARTIAL_MAX` | 1.0 | Adaptive TP1: max sell fraction (1.0 = full exit at TP1) |
| `TP1_CLOSE_DIST` | 80.0 | Adaptive TP1: considered "close to strike" when `abs(distance) <= this` |
| `TP1_LATE_MIN_REM` | 8.0 | Adaptive TP1: considered "late" when minutes remaining <= this |
| `TP1_POST_DROP_THRESH` | 0.05 | Adaptive TP1: posterior drop from entry (pp) to treat as degraded |
| `TP1_EDGE_DROP_THRESH` | 0.005 | Adaptive TP1: edge drop from entry to treat as degraded |
| `TP1_TRAIL_PRICE_ACTIVATION_PCT` | 0.02 | Post-TP1 tighter trailing: arm once MFE exceeds this |
| `TP1_TRAIL_PRICE_DISTANCE_PCT` | 0.06 | Post-TP1 tighter trailing: trail distance from peak price |
| `LATE_CONVICTION_POSTERIOR` | 0.80 | Min posterior for late-window sniping |
| `LATE_CONVICTION_DISTANCE` | 40.0 | Min BTC distance from strike for late sniping |
| `LATE_CONVICTION_DISTANCE_ATR_MULT` | 0.25 | Dynamic ATR-scaled fallback for late sniping distance |
| `LATE_CONVICTION_MIN_SCORE` | 1.25 | Strict minimum score floor for late sniping overrides |
| `MIN_ENTRY_DISTANCE_ATR_MULT` | 0.25 | Global ATR-scaled distance requirement for entry |
| `BB_SQUEEZE_NORMAL` | 0.0030 | BB squeeze gate threshold (normal regime) |
| `DAILY_LOSS_LIMIT_PCT` | 0.10 | Daily loss limit as fraction of session start balance |
| `LATE_WINDOW_FOK_MIN_REM` | 4.0 | Force FOK (not GTC) for entries < 4 min remaining |
| `ONE_SIDED_CONFIRM_CYCLES` | 2 | Require 2+ consecutive cycles of one-sided clearance |
| `FORCED_DRAWDOWN_GRACE_SEC` | 15.0 | Grace period: don't fire FORCED_DRAWDOWN in first 15s |
| `PUMP_REVERSION_THRESHOLD` | 0.05 | 5% single-cycle pump triggers limit-below entry |
| `PUMP_REVERSION_OFFSET` | 0.03 | Buy $0.03 below mid on pump detection |
| `MAX_SELL_ATTEMPTS_PER_CYCLE` | 2 | Max sell HTTP calls per 5s cycle (caps RUNTIME) |
| `MAX_CONSECUTIVE_SELL_FAILURES` | 20 | After this many failures, stop trying — wait for auto-settle |
| `MIN_SELL_SIZE` | 0.05 | Minimum sell size in shares; below this, write off as dust instead of placing order |
| `MOMENTUM_GATE_ATR_THRESHOLD` | 0.25 | ATR-normalized 15s BTC velocity magnitude to block adverse entries |
| `PM_LOB_ADVERSE_THRESHOLD` | 0.80 | PM book ask-ratio above this blocks entry (distribution signal) |
| `FUNDING_RATE_GATE_THRESHOLD` | 0.0002 | Funding rate opposing direction above this blocks entry (~0.02%/8h) |
| `MIN_ENTRY_DISTANCE_ATR_MULT` | **0.40** | ATR-scaled distance requirement (raised from 0.25 — Fix #2) |
| `BAYES_SIGNAL_WEIGHT` | **0.30** | Bayesian model weight in Bayesian-market blend (market weight = 0.70 — Fix #4) |
| `HTF_TREND_GATE_ENABLED` | `true` | Enable/disable 1H EMA9/EMA20 trend conflict gate (Fix #6) |
| `VPOC_PROXIMITY_GATE_PCT` | **0.015** | Max allowed % deviation from VPOC before blocking entry (Fix #9, 1.5%) |
| `EMERGENCY_SELL_LOSS_PCT` | **0.08** | Emergency FOK market-sell threshold when first sell failed and loss > 8% (Fix #8) |

