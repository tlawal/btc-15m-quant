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

### April 2 Audit Improvements
| Fix | Change | File |
|-----|--------|------|
| **#1** | **Late-Entry Gate** — rejects positions with < 1.5 minutes remaining. | `main.py` |
| **#2** | **Partial Fill Tracking** — accurately carries forward partial balances out of failed entry orders. | `main.py` |
| **#3** | **Sub-3 Minute Exit Suppression** — completely suppresses Layer 5 exits near expiry if the model conviction is high (posterior > 0.85) AND the underlying BTC price confirms the direction. Resolves noise-induced panic selling on winning trades. | `exit_policy.py` |
| **#4** | **Bid-Side Collapse Detection (MM Bait)** — suppresses Layer 5 exits for 60s if the bid price suddenly drops >20% below our entry bid or the spread widens >15%. Survives localized order-book shocks. | `exit_policy.py` |

### April 3 Audit Findings & Fixes
**Problem**: 1 catastrophic loss (-38.9%) on a winning trade (12:41 PM, NO @ $0.87 sold @ $0.54, market resolved at $1.00), plus 3/3 other wins had premature exits cutting profits early. Total money left on table: $4.07 across 6 trades.

**Root causes**:
- **MODEL_REVERSAL fired after 21s** on a monster-signal entry due to posterior drop from CLOB liquidity noise, not genuine model failure
- **REVERSE_CONVERGENCE exited winning positions** because opposing bid spiked transientally; didn't check if held position was profitable
- **TypeError in sizing.py** silently crashing position-size computation

**Fixes implemented (Apr 3, 2026)**:
| Fix | Change | File |
|-----|--------|------|
| **#A** | **MODEL_REVERSAL monster suppression** — suppress exit when entry score ≥ 8.0 OR entry_posterior ≥ 0.90 AND minutes_remaining < 5.0. Near-expiry CLOB noise dominates; hold to settlement. | `exit_policy.py`, `config.py` |
| **#B** | **REVERSE_CONVERGENCE profit guard** — only fire when held position is losing (unrealized < -2%). Opposing bid spikes on thin books are noise. | `exit_policy.py` |
| **#C** | **Sizing robustness** — None guard before SLIPPAGE_BUFFER multiplication to prevent TypeError crash. | `sizing.py` |

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
- **Runner protection after TP1** — after TP1 fills, remainder is protected via:
  - `TP1_TRAIL_PRICE_STOP` (tighter price-based trailing)
  - `RUNNER_POSTERIOR_COLLAPSE` (exit remainder if posterior collapses from entry by `RUNNER_POSTERIOR_DROP_PCT` while still in profit by `RUNNER_MIN_PROFIT_PCT`).
- **MODEL_REVERSAL monster suppression (Apr 3, 2026)** — when a monster signal (score ≥ 8.0 OR posterior ≥ 0.90) is entered with < 5 min remaining, suppress MODEL_REVERSAL exits entirely. Near expiry, CLOB spreads widen 20-30%, making posterior drops reflect liquidity dynamics, not genuine belief changes. Holding to settlement is +EV. Gate: `MODEL_REVERSAL_MONSTER_SUPPRESS_MIN_REM = 5.0`.
- **REVERSE_CONVERGENCE profit guard (Apr 3, 2026)** — only fire REVERSE_CONVERGENCE when the held position is losing (unrealized < -2%). Opposing-side bid spikes in thin markets are transient noise; selling a profitable position because the opponent's bid spiked destroys edge. Prevents profit-taking too early on winning trades.

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
6. **Biais, B., Hillion, P., & Spatt, C. (1995). "An Empirical Analysis of the Limit Order Book and the Order Flow in the Paris Bourse." _Journal of Finance_, 50(5).** (CLOB spread widening near events is market-maker inventory management, driving our bid-pull / MM bait detection logic)

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
- **Dip-Catching Limit Algorithm** — as the default behavior, the bot sets the entry order conditionally based on the order book. By default it sits on the passively safe `bid + $0.01` tick to ensure it enters cautiously, relying on limit orders rather than crossing the spread natively (saving on fees). It waits up to 30 seconds for price oscillations to fill the limit bid before escalating.
- **FOK vs passive limit decision framework** — monster signals or late-window (<4 min) → FOK at ask; pump detected → limit below mid; normal → passive `bid + tick`. Based on Cont & Kukanov (2017): aggressive fills when alpha is strong + time is short; passive when signal is moderate + time permits.
- **Data-driven offset optimization (planned)** — optimizer.py will log fill rates and entry conditions, and tune `PUMP_REVERSION_OFFSET` adaptively based on historical fill probability vs. edge captured. Per Gueant, Lehalle & Fernandez-Tapia (2013), the optimal offset from fair value is larger in thin markets (low order arrival intensity).

### Execution
- **Smart entry pricing** — passive `bid+1tick` for GTC, aggressive `ask` for FOK monster signals, reversion `mid - offset` on pump detection
- **Mid-price fallback** — when ask is None (deep ITM), uses `mid + 0.01` capped at 0.99
- **Depth-aware sizing** — caps position at 50% of top-of-book depth
- **Stale order replace** — cancel + re-place after 12s if entry GTC not filled
- **Reprice safety check (entry orders)** — caps stale-entry replacements (`REPRICE_MAX_COUNT` + `REPRICE_MIN_INTERVAL_SEC`) and cancels instead of chasing if the replacement would worsen price beyond `REPRICE_MAX_WORSEN_PCT` (when `REPRICE_CANCEL_ON_WORSEN=true`).
- **Exit timeout FOK** — if exit order pending > 60s, force-replaces as FOK at `bid - 1 tick`
- **Price surge take-profit** — if price jumps by `PRICE_SURGE_TRIGGER_PCT` within `PRICE_SURGE_WINDOW_SEC` (and overall unrealized >= `PRICE_SURGE_MIN_PROFIT_PCT`), force an immediate take-profit (`PRICE_SURGE_TAKE_PROFIT`).
- **Close-on-expiry** — if a position remains on an old market close to expiry (`CLOSE_ON_EXPIRY_MIN_REM`), force a `CLOSE_ON_EXPIRY` exit instead of relying on settlement.
- **FOK immediate reconcile recheck** — for FOK exits, the engine re-checks order status after a short delay (`FOK_RECHECK_DELAY_SEC`) before concluding the order was killed (eventual-consistency hardening).
- **State checkpoint** — saves state before every order placement (crash-safe)
- **Startup reconciliation** — checks pending orders AND live API positions on restart; populates `held_position` from Polymarket if local state is out of sync
- **On-chain position reconciliation** — on startup and before every sell, queries on-chain ERC-1155 `balanceOf` (Gnosis CTF contract) to verify inherited position size is correct. Clears phantom positions (on-chain balance = 0) immediately. Catches stale Polymarket positions-API data that can cause futile sell loops.
- **Sell retry cap** — limits sell attempts to `MAX_CONSECUTIVE_SELL_FAILURES` (default 20) total; after that, stops retrying and waits for auto-settle. Prevents 3+ minute loops of futile HTTP calls when shares don't exist on-chain.
- **Sell failure diagnostics** — on sell failure, logs `SELL_FAIL_DEBUG` with positions-API response and on-chain balance side-by-side for root cause analysis.
- **Full-position sell rounding** — `limit_sell` uses `round(size)` (nearest integer) instead of `int(size)` (floor) for sizes > 1 share. `int(6.9905)` silently dropped 0.9905 shares per partial exit, leaving a rump that then triggered a separate HARD_STOP exit at a worse price.
- **Dust write-off guard** — if `sell_size < MIN_SELL_SIZE (0.05 shares)` after all on-chain/API clamping, the position is written off as `DUST_WRITEOFF` with blended PnL rather than placing an order that will fail with a 400 allowance error.
- **Fractional Dust Eradication** — `pos.size` subtractions on partial fills are strictly subjected to `math.floor(size * 10000) / 10000`, matching Polymarket's token precision limit natively and completely neutralizing recursive `DUST_SKIP` 82,000% PNL UI math bugs.
- **Blended PnL on auto-settle** — `AUTO_SETTLE_WIN` and `AUTO_SETTLE_LOSS` now compute `blended_exit_price` and `blended_pnl` weighted across all partial exits plus the remaining shares at settlement ($1.00 or $0.00). Previously, auto-settle overwrote `pnl = -1.0` even when 90%+ of the position had already been sold at recovery prices, causing the dashboard to show -100% instead of the real ~-15%. If an older trade was misclassified, use `POST /api/repair-trade` instead of resetting the database.
- **Slippage tracking** — actual fill price vs intended price logged per trade; warns if > 1%
- **Market quality filter** — skips cycle if spreads > 8%, book depth < 20 USDC, or klines > 5 min stale
- **Position sizing robustness (Apr 3, 2026)** — added None guard in `compute_position_size()` before SLIPPAGE_BUFFER multiplication to prevent TypeError crashes when balance or Kelly fraction becomes None

**Implemented:**
- **Critical-exit FOK fallback ladder** — for critical exits (e.g., hard-stops / strike-distance / close-on-expiry), if the initial FOK sell fails, the bot retries up to 2 additional FOK attempts crossing deeper: `bid`, then `bid - 1 tick`, then `bid - 2 ticks`.
- **Critical-exit cycle-lag bypass** — critical exits bypass `CYCLE_LAG_SELL_SKIP` so protective exits are not delayed by a slow cycle.
- **Pump reversion entry pricing** — when `PUMP_DETECTED`, entries are priced below the pumped mid (`mid - PUMP_REVERSION_OFFSET`) rather than at/above the bid.
- **Pump cool-off** — after `PUMP_DETECTED`, the bot blocks new entries for `PUMP_COOLOFF_SEC` seconds (unless `monster`).
- **Distance-aware entry filter** — blocks entries when `|Dist|/ATR5m > DIST_ENTRY_MAX_ATR_RATIO` (unless `monster`).

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

Test note:
- `test_redeem.py` includes a version-compatible POA middleware import to support newer `web3` releases.

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
| `/api/repair-trade` | POST | Admin-token protected: repairs a misclassified historical trade in place (body: `{market_slug, dry_run}`) |
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

## Forensic Trade Audit (Live)

All state that matters for a post-mortem lives on the Railway volume at `/data/state.db`, not the local repo copy. Use the dashboard admin endpoints to pull ground truth and reconstruct the exit-decision timeline. All endpoints below are **read-only**.

### Prerequisites
- `railway` CLI authenticated against the `pleasing-delight` project
- Admin token from `railway variables | grep DASHBOARD_ADMIN_TOKEN`
- Public URL from `railway variables | grep RAILWAY_PUBLIC_DOMAIN` (e.g. `btc-15-quant.up.railway.app`)

### 1. Pull the live state DB
```bash
ADMIN_TOKEN=qosmio9176   # or read from: railway variables
BASE=https://btc-15-quant.up.railway.app
curl -sSL "$BASE/api/db/download" -H "x-admin-token: $ADMIN_TOKEN" -o /tmp/live_state.db
```

### 2. Query the trade(s) of interest
```bash
python3 - <<'PY'
import sqlite3, json
c = sqlite3.connect('/tmp/live_state.db').cursor()

# List recent closed trades (id, time, slug, sizes, prices, exit reason)
print("=== closed_trades (recent) ===")
for r in c.execute(
    "SELECT id, datetime(timestamp,'unixepoch','localtime'), market_slug, "
    "size, entry_price, exit_price, pnl_usd, exit_reason "
    "FROM closed_trades ORDER BY timestamp DESC LIMIT 20"
):
    print(r)

# Full row + entry features for a specific market slug
TARGET = "btc-updown-15m-1775823300"   # <-- edit
print("\n=== closed_trades row + features for", TARGET, "===")
for r in c.execute("SELECT * FROM closed_trades WHERE market_slug=?", (TARGET,)):
    row = dict(zip([d[0] for d in c.description], r))
    if row.get("features"):
        try: row["features"] = json.loads(row["features"])
        except Exception: pass
    print(json.dumps(row, indent=2, default=str))

# trade_history kv has the full partial_exits lineage per trade
print("\n=== trade_history partial_exits ===")
c.execute("SELECT value FROM kv WHERE key='trade_history'")
for item in json.loads(c.fetchone()[0]):
    if TARGET.endswith(str(item.get("window",""))):
        print(json.dumps(item, indent=2))
PY
```

The `partial_exits` list in `trade_history` is the single most valuable artifact — each item shows `{reason, price, size, ts}` for every exit leg. A sequence like `TP2 → HARD_STOP → DUST_WRITEOFF` tells the complete story of a losing runner.

### 3. Pull aggregate context endpoints
```bash
curl -s "$BASE/api/optimizer-detail"   -H "x-admin-token: $ADMIN_TOKEN" | python3 -m json.tool
curl -s "$BASE/api/exit-stats"         -H "x-admin-token: $ADMIN_TOKEN" | python3 -m json.tool
curl -s "$BASE/api/regime-performance" -H "x-admin-token: $ADMIN_TOKEN" | python3 -m json.tool
curl -s "$BASE/api/debug"              -H "x-admin-token: $ADMIN_TOKEN" | python3 -m json.tool | head -80
```

Red flags to scan for:
- `/api/debug.last_cycle_error` — any active runtime exception bubbling out of `_cycle()` / `_handle_entry()` / `_handle_exits()`
- `/api/optimizer-detail.score_offset` or `.edge_offset` **at their floors** (-1.0 / -0.01) — anti-learning loop: the optimizer is overfitting on a tiny in-sample and relaxing entry gates to the max
- `/api/exit-stats.by_reason.TRAIL_PRICE_STOP.avg_unrealized_pct` **negative** — trailing stop is firing on already-losing trades instead of locking profits
- `/api/exit-stats.by_reason.HARD_STOP.count` dominating — protective exits are failing upstream, leaving `HARD_STOP` to clean up

### 4. Tail live logs (short retention)
```bash
(railway logs 2>&1 & sleep 15; kill $!) > /tmp/rail.log
grep -E "TARGET_SLUG|TRAIL_PRICE_STOP|HARD_STOP|DUST|TP[123]|TP_FULL|EXIT" /tmp/rail.log
```
Railway only keeps ~minutes of logs on the `railway logs` stream, so for older trades the DB `partial_exits` lineage is the authoritative source.

### 5. Prompt template for Claude Code forensic auditor

Paste the following into Claude Code whenever a trade needs a forensic audit:

> Run a forensic audit on the losing trade `<market_slug>` (approximately `<HH:MM>`, `<side>` entry `<price>`, observed PnL `<pct>%`).
>
> 1. `curl -sSL https://btc-15-quant.up.railway.app/api/db/download -H "x-admin-token: $ADMIN_TOKEN" -o /tmp/live_state.db`, then query `closed_trades` for that market slug and extract the matching item from `kv.trade_history`. Print every field, parse `features`, and show every `partial_exits` entry in order with their reason/price/size/ts.
> 2. From `/api/debug` report `last_cycle_error` and any active runtime bugs.
> 3. From `/api/optimizer-detail` report `score_offset`, `edge_offset`, `last_precision`, `total_features_logged`, `closed_trades`.
> 4. From `/api/exit-stats` list the top exit reasons by count with their `avg_unrealized_pct`.
> 5. Read `exit_policy.py`, `main.py _handle_exits`, and `config.py` and reconstruct the exact exit-decision timeline: for each `partial_exits` entry, identify which exit layer fired (e.g. `TP2`, `HARD_STOP`, `VOL_HARD_STOP`), which config thresholds it crossed, and cite `file.py:line`.
> 6. Identify root causes — both the *strategic* failure (wrong rule fired / no rule fired) and the *execution* failure (maker sells on a collapsing book, dust residuals, etc.).
> 7. Propose concrete, surgical fixes — config edits or code edits cited to `file.py:line` with a one-line *why* each. No generic "add logging" suggestions.
> 8. Finish with a one-paragraph summary of what the user saw vs what the bot actually did, and name the single highest-impact fix.
>
> Only use read-only tools. Do not modify any files.

---

## FAQ

**Why does the dashboard show the wrong win rate or an `AUTO_SETTLE_LOSS` on a trade that ended profitable?**  
That usually means a historical settlement was classified before the blended PnL fix. Do not reset the database unless you want to lose trade history. Use the admin-protected `POST /api/repair-trade` endpoint to patch the bad trade in place and recompute metrics.

**What does the repair endpoint do?**  
It finds the matching historical trade, rewrites the canonical `trade_history` entry, updates the corresponding `closed_trades` row, removes duplicates created by older settlement logic, and recalculates performance counters.

**When should I use `/api/db/reset`?**  
Only when you intentionally want a clean slate. It deletes all stored trades and state, so it is not the right fix for a single bad settlement.

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
