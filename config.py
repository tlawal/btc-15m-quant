import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
load_dotenv()


class Config:
    # ── Timing ────────────────────────────────────────────────────────────────
    LOOP_INTERVAL_SEC          = 3        # inner loop cadence (3s: faster reversal detection; well within Binance rate limits)
    WINDOW_SEC                 = 900      # 15-minute binary window
    EARLY_WINDOW_GUARD_MIN     = 7.5     # block non-monster trades in first 4 min (entries only in last 7.5 min)
    BELIEF_VOL_LOOKBACK_SEC    = 180      # rolling σ_B window (3 min)
    BELIEF_VOL_DEFAULT         = 0.15
    BELIEF_VOL_MULT_MIN        = 0.70
    ENSEMBLE_BAYES_WEIGHT      = 0.40
    ENSEMBLE_MODEL_WEIGHT      = 0.60
    BASIS_EDGE_MIN             = 0.008
    HAWKES_LATE_REQUIRED_EDGE  = 0.003
    OUTSIDE_HOURS_ENTRY_POSTERIOR_MIN = 0.92
    # Belief-volatility multiplier caps: keep regression toward 0.5 gentle, not crushing.
    BELIEF_VOL_MULT_MAX        = 1.30     # normal cap when more than 5 min remain
    BELIEF_VOL_LATE_MAX        = 1.15     # tighter cap when < 5 min remain
    # Optional: scale belief-volatility effect by time remaining in the window.
    # When enabled, high σ_B early in the candle has more influence than very late.
    BELIEF_VOL_TIME_DECAY_ENABLED = False

    # ── Dashboard admin controls ─────────────────────────────────────────────
    # If set, dashboard can issue authenticated manual commands (e.g. manual exits).
    # Must be provided via env var on the server.
    DASHBOARD_ADMIN_TOKEN      = os.getenv("DASHBOARD_ADMIN_TOKEN", "")

    # ── Score → posterior coupling (optional) ──────────────────────────────────
    # Allows a very strong signed_score to gently tilt the Bayesian posterior.
    # This is deliberately small and off by default; downstream consumers can
    # inspect both raw and adjusted posteriors.
    SCORE_POSTERIOR_COUPLING_ENABLED = False
    SCORE_POSTERIOR_MAX_SHIFT        = 0.03   # max absolute probability shift
    SCORE_POSTERIOR_COUPLING_K       = 0.01   # scale applied to (score / 10)

    # ── Regime-adaptive thresholds (FIX #7) ──────────────────────────────────
    # ATR regime boundaries (5m ATR on BTC in USD)
    ATR_LOW_THRESHOLD          = 80.0
    ATR_HIGH_THRESHOLD         = 200.0
    # Min score by regime
    MIN_SCORE_LOW_VOL          = 3.0
    MIN_SCORE_NORMAL           = 2.5
    MIN_SCORE_HIGH_VOL         = 2.0
    # Required edge by regime
    REQUIRED_EDGE_LOW          = 0.050
    REQUIRED_EDGE_NORMAL       = 0.005   # lower edge req in normal regime → higher trade frequency
    REQUIRED_EDGE_HIGH         = 0.025
    # Balance-adaptive edge: when wallet is small, use lower edge requirement
    # This prevents the bot from being perpetually blocked at low capital.
    LOW_BALANCE_THRESHOLD_USD  = 30.0   # apply lower edge when balance < this
    REQUIRED_EDGE_LOW_BALANCE  = 0.007  # relaxed edge requirement at low balance (was 0.020 — too high for sub-$20)
    # Fallback ATR used when local ATR computation is unavailable.
    DEFAULT_ATR                = 150.0

    # ── Late-window conviction override ──────────────────────────────────────
    # When near expiry with strong posterior and BTC clearly away from strike,
    # technical scores (EMA/MACD) are stale — the market has already repriced.
    # Suppress the score gate and relax the edge gate so the bot can snipe.
    # Tuned from log analysis: the window 1772984700 had posterior 0.82–0.91,
    # distance +40, edge 4–16%, score −0.8 to −1.9 → all missed due to score_low.
    LATE_CONVICTION_MIN_REM    = 3.0      # must be within last 3 min of window
    LATE_CONVICTION_POSTERIOR  = 0.80     # model must be ≥80% confident (was 0.93, too strict)
    LATE_CONVICTION_DISTANCE   = 40.0     # price must be ≥$40 from strike
    LATE_CONVICTION_EDGE       = 0.006    # relaxed edge requirement (was 0.020 — too high for near-certain late trades)
    LATE_CONVICTION_DISTANCE_ATR_MULT = 0.25 # dynamic ATR-based distance requirement
    LATE_CONVICTION_MIN_SCORE  = 1.25     # strict safety floor to prevent pure noise entries
    LATE_CONVICTION_MAX_ENTRY_PRICE = 0.95  # Audit 4: cap entry price — paying >$0.95 gives <5% upside vs catastrophic slippage risk near expiry
    MIN_ENTRY_DISTANCE_ATR_MULT = 0.50    # Audit 2 P2: raised from 0.40 — blocks noise-range entries (<0.5 ATR from strike)

    # ── ADX trend filter (FIX #5) ─────────────────────────────────────────────
    ADX_TREND_THRESHOLD        = 20.0     # below = choppy, block directional entry

    # ── OFI / microstructure ─────────────────────────────────────────────────
    OFI_STRONG                 = 5.0
    OFI_15M                    = 15.0
    VPIN_BLOCK_THRESHOLD       = 0.85     # block entry if vpin > this AND score weak (raised: fewer false halts)

    # ── VPIN toxicity regimes (exit/hold shaping) ─────────────────────────────
    VPIN_TOXIC_THRESHOLD       = 0.85     # VPIN above this = toxic flow regime
    VPIN_TOXIC_HOLD_MAX_SEC    = 240.0    # shorten holding in toxic regime (theta + adverse selection)

    OUTSIDE_HOURS_TAKE_PROFIT_PCT = 0.02

    # ── Scoring constants ─────────────────────────────────────────────────────
    BLIND_ENTRY_SCORE          = 7.0      # legacy; use min_score_* in practice
    MONSTER_SCORE              = 8.0
    MONSTER_POSTERIOR          = 0.90
    STOP_LOSS_DELTA            = 7.0

    MIN_HOLD_SECONDS           = 30
    STOP_LOSS_CONSEC_CHECKS    = 3
    STOP_LOSS_PERSIST_SECONDS  = 12
    STOP_LOSS_COOLDOWN_SECONDS = 120

    # ── Entry Strategy ────────────────────────────────────────────────────────
    CONSERVATIVE_ENTRY_OFFSET  = 0.02     # 2-tick discount on standard entries (Fix #8)
    STRICT_EDGE_GATE_ENABLED   = True     # block entries if target_edge <= 0 (unless monster)
    
    # ── Exit parameters ───────────────────────────────────────────────────────
    TAKE_PROFIT_PRICE          = 0.99
    SLIPPAGE_BUFFER_PCT        = 0.008    # Phase 6: 80bps execution buffer
    
    # ── Micro-Exit Sensitivity Shaping (Fix #8) ──────────────────────────────
    # Factor to multiply microstructure thresholds (OFI/CVD) during early trade phase.
    # Higher value = HIGHER thresholds = LESS sensitive (harder to trigger).
    # We use 2.0 to make thresholds 2x harder to trigger when > 5 min remain.
    # This prevents 'paper-handing' on minor noise while the trade has time to run.
    MICRO_EXIT_MID_WINDOW_SENSITIVITY = 2.0
    MAX_DRAWDOWN_PCT           = 0.15    # Widened: let winners run, TRAIL_PRICE_STOP handles profit protection
    HARD_STOP_PCT              = 0.25    # HARD unconditional circuit breaker — no posterior gate.
                                          # Trailing guard held a -65% loss when posterior lagged price.
                                          # At -25% the position is unrecoverable on a 15m binary; cut always.
    # Posterior-gated grace period for HARD_STOP (Audit 3 P3):
    # When posterior still supports the position, delay HARD_STOP to filter microstructure noise.
    HARD_STOP_GRACE_SEC            = 45    # grace when posterior >= HIGH threshold
    HARD_STOP_GRACE_POST_HIGH      = 0.70  # posterior above this → full grace
    HARD_STOP_GRACE_POST_LOW       = 0.50  # posterior between LOW and HIGH → reduced grace
    HARD_STOP_GRACE_REDUCED_SEC    = 25    # grace when posterior between LOW and HIGH
    FORCED_LATE_EXIT_MIN_REM    = 1.0     # Force exit at <1 min if losing
    FORCED_DISTANCE_EXIT_MIN_REM = 3.0
    FORCED_PROFIT_LOCK_MIN_REM = 1.0
    FORCED_LATE_LOSS_PCT       = 0.10   # tightened from 0.15 — cut losers faster in last 5 min
    FORCED_PROFIT_PCT          = 0.25
    FORCED_DISTANCE_MAX        = 30.0     # abs(btcPrice - strike) < this triggers late exit

    MID_WINDOW_PROTECTIVE_LOSS_PCT = 0.05
    MID_WINDOW_POSTERIOR_CEIL      = 0.55

    # ── MAE (Maximum Adverse Excursion) framework ────────────────────────────
    MAE_TIGHTEN_THRESHOLD          = 0.10   # MAE >= 10% → tighten drawdown tolerance 40%
    MAE_RECOVERY_EXIT_THRESHOLD    = 0.20   # MAE >= 20% → recovery exit when near entry
    MAE_DEEP_THRESHOLD             = 0.25   # MAE >= 25% → maximum pain, hard exit
    MAE_RECOVERY_NEAR_ENTRY_PCT    = 0.03   # "near entry" = within 3% of entry price
    MAE_RECOVERY_TIME_LATE_SEC     = 300    # 5 min: after this, accept -10% instead of -3%

    # ── Late-Window Entry Hardening ─────────────────────────────────────────
    LATE_WINDOW_FOK_MIN_REM        = 4.0    # Force FOK (not GTC) for entries < 4 min remaining
    ONE_SIDED_CONFIRM_CYCLES       = 2      # Require 2+ consecutive cycles of one-sided clearance
    MIN_HOLD_BEFORE_DRAWDOWN_SEC   = 30     # Base grace — scaled down near expiry (Audit Change 2)
    SCORE_STABILITY_MIN_CYCLES     = 3      # Require score above threshold for 3+ consecutive cycles (IC stability gate)
    MAX_ENTRY_PRICE_GTC            = 0.92   # Block GTC entries above this price (tiny upside, huge downside risk)
    STREAK_POSTERIOR_MIN_BOOST     = 0.05   # Extra posterior required after 2+ consecutive losses

    # ── Sell Retry Limits ─────────────────────────────────────────────────────
    MAX_SELL_ATTEMPTS_PER_CYCLE    = 2      # Max sell attempts per 5s cycle (caps RUNTIME)
    MAX_CONSECUTIVE_SELL_FAILURES  = 20     # After this many, stop trying and let auto-settle
    MIN_SELL_SIZE                  = 0.05   # Shares — below this CLOB rejects; write off as dust

    # ── Entry protection gates ────────────────────────────────────────────────
    # BTC momentum velocity gate: block entry when BTC surges against trade direction.
    # Threshold in ATR-normalized units over 15s (3 cycles × 5s).
    MOMENTUM_GATE_ATR_THRESHOLD    = 0.25   # e.g. 0.25 = BTC moved 25% of ATR in 15s adversely
    # Polymarket LOB adverse imbalance gate: heavy ask side = distribution signal.
    PM_LOB_ADVERSE_THRESHOLD       = 0.80   # ask_size / total_book > this = adversely imbalanced
    # Cross-asset funding rate gate: positive funding opposes DOWN trade (bullish bias).
    FUNDING_RATE_GATE_THRESHOLD    = 0.0002 # ~0.02% per 8h — strong directional funding bias
    # PM price surge gate: block entry if target side surged >X% in one cycle (convergence consumed).
    PM_PRICE_SURGE_GATE            = 0.12   # 12% single-cycle surge = opportunity already passed
    # Convergence snipe: bypass not_one_sided when model leads market by >= gap and BTC positioned.
    CONVERGENCE_GATE_POSTERIOR_MIN = 0.82   # minimum posterior to allow convergence bypass
    CONVERGENCE_GATE_GAP_MIN       = 0.12   # minimum model-market price gap to trigger snipe
    # Minimum minutes remaining for entry: 3.0 for normal, 2.0 for near-certain (post >= 0.90).
    MIN_ENTRY_MINUTES_REM          = 3.0    # raised from 2.0 — late entries have no recovery time
    # Sell escalation: market_sell after N consecutive 400-balance errors (was 10 — too slow).
    # R7: faster escalation to market_sell on failed maker legs. Two consecutive
    # maker failures on a collapsing book is already enough dust to risk DUST_WRITEOFF.
    ALLOWANCE_FAIL_MARKET_ESCALATION = 1    # escalate to market_sell after 1 failure
    # Late-window position sizing: 50% reduction for entries with < 3 min remaining (non-monster).
    LATE_WINDOW_SIZE_MULTIPLIER    = 0.50   # Kelly calibrated for 15m; at <3m variance is maximal
    # Cycle latency watchdog: skip sell if cycle has been running longer than this to avoid collisions.
    CYCLE_LAG_SELL_SKIP_SEC        = 2.5    # 2.5s = half of 5s baseline cycle

    # ── Pump Reversion Entry ────────────────────────────────────────────────
    PUMP_REVERSION_THRESHOLD       = 0.05   # 5% single-cycle pump triggers limit-below entry
    PUMP_REVERSION_OFFSET          = 0.03   # buy $0.03 below mid on pump detection
    PUMP_COOLOFF_SEC               = 15     # after a pump, block new entries briefly to avoid chasing

    # ── Transition-zone gate stability (Audit 3 P2) ─────────────────────────
    # For entries in the transition zone (first 9 min elapsed = 6.0-7.5 min remaining),
    # require N consecutive all-clear gate cycles before entering.
    # Filters entries that only clear on transient signal fluctuations.
    TRANSITION_ZONE_MIN_REM_LOW    = 6.0    # transition zone lower bound (min remaining)
    TRANSITION_ZONE_MIN_REM_HIGH   = 7.5    # transition zone upper bound (= early_window_guard)
    TRANSITION_GATE_STABLE_CYCLES  = 2      # require 2 consecutive all-clear cycles

    # ── FOK-only policy in early window (Audit 3 P4) ─────────────────────
    # After MAX_FOK_ATTEMPTS_EARLY FOK failures with > FOK_ONLY_MIN_REM_THRESHOLD
    # min remaining, skip entry entirely — no GTC fallback.
    # Academic basis: Glosten-Milgrom, Griffiths et al., Kyle, Almgren-Chriss.
    FOK_ONLY_MIN_REM_THRESHOLD     = 3.0    # GTC fallback only allowed with ≤3 min remaining
    MAX_FOK_ATTEMPTS_EARLY         = 2      # after 2 FOK failures in early window, skip entry

    # ── Distance-aware entry (trend-extension filter) ───────────────────────
    # Block entries when BTC is already far from strike in ATR units; these moves
    # are prone to mean reversion back toward strike.
    DIST_ENTRY_MAX_ATR_RATIO       = 0.60

    # ── Critical-exit execution fallback (FOK ladder) ───────────────────────
    # For critical exits, retry FOK sells crossing deeper if full-fill fails.
    CRITICAL_EXIT_FOK_LADDER_ENABLED = True
    CRITICAL_EXIT_FOK_LADDER_STEPS   = 3      # bid, bid-1tick, bid-2tick
    CRITICAL_EXIT_FOK_LADDER_TICK    = 0.01   # Polymarket min price increment

    # ── Reprice safety (stale order replacement guardrails) ─────────────────
    # Prevent stale-entry replacement from "chasing" the market too far from the
    # original intended price.
    REPRICE_MAX_COUNT              = 2
    REPRICE_MIN_INTERVAL_SEC       = 6
    # For BUY (entry) orders, never worsen the intended price by more than this.
    REPRICE_MAX_WORSEN_PCT         = 0.03
    # If new price would worsen beyond threshold, cancel the entry instead of chasing.
    REPRICE_CANCEL_ON_WORSEN       = True
    # Audit 2 P3: short-lived GTC fallback after FOK failures — hard cap on pending time.
    GTC_MAX_PENDING_SEC            = 10

    # ── Price surge handling (fast-move profit capture) ─────────────────────
    # If price surges quickly in our favor, take profit early (prevents missing
    # brief spikes / wick moves).
    PRICE_SURGE_WINDOW_SEC         = 15
    PRICE_SURGE_TRIGGER_PCT        = 0.12
    PRICE_SURGE_MIN_PROFIT_PCT     = 0.05
    PRICE_SURGE_EXIT_REASON        = "PRICE_SURGE_TAKE_PROFIT"

    # ── Close-on-expiry (remainder) ─────────────────────────────────────────
    # If position is still open on the old market near expiry, force-close rather
    # than relying on auto-settle.
    CLOSE_ON_EXPIRY_MIN_REM        = 0.75

    # Volatility-adjusted trailing stop (probability-space, ATR-scaled)
    TRAIL_ATR_REF              = 120.0    # reference ATR level for scaling trailing width
    TRAIL_BASE_POST_DROP       = 0.04     # base allowable posterior drop from peak before exit
    TRAIL_ATR_SCALE            = 0.06     # additional allowed drop at high ATR (scaled by atr/ref)
    TRAIL_MIN_POST_DROP        = 0.02     # floor (tightest) trailing width
    TRAIL_MAX_POST_DROP        = 0.12     # cap (widest) trailing width
    TRAIL_ARM_MIN_PROFIT_PCT   = 0.015    # only arm trailing after at least +1.5% unrealized
    TRAIL_MIN_HOLD_SEC         = 30.0     # minimum hold before trailing can trigger

    # ── Price-Based Trailing Stop ─────────────────────────────────────────────
    # R4: Tightened from 10% → 4% give-back. Exit-stats showed TRAIL_PRICE_STOP
    # firing at avg −13.83% unrealized (i.e. AFTER profit already round-tripped to a
    # loss). A 4% give-back on an 8% peak locks in +4%; the old 10% distance never
    # protected against fast strike-flip reversions.
    TRAIL_PRICE_ACTIVATION_PCT = 0.06     # arm only after a real 6% win
    TRAIL_PRICE_DISTANCE_PCT   = 0.04     # trail 4% behind highest price

    # ── Tiered Take-Profit (percentage from entry) ─────────────────────────
    TP1_PCT                    = 0.05     # +5% → sell 1/3 (or full if partial disabled)
    TP2_PCT                    = 0.12     # +12% → sell 1/3
    TP3_PCT                    = 0.20     # +20% → sell remaining
    # R1: extend late-entry single-TP to cover the 0.88–0.95 "dead zone" that
    # used to fall between TP1 (+5% unreachable from 0.92 before strike-flip) and
    # the old TP_LATE_ENTRY @ 0.95. Target is linearly scaled so cheap-late entries
    # still target more upside while near-cap entries take tight exits.
    TP_LATE_ENTRY_THRESH       = 0.88     # entry >= $0.88 triggers single-tier TP
    TP_LATE_ENTRY_PCT          = 0.02     # floor target (applied at entry ~0.95)
    TP_LATE_ENTRY_PCT_MAX      = 0.04     # ceiling target (applied at entry ~0.88)
    TP_LATE_ENTRY_BAND_TOP     = 0.95     # entries at/above this hit the floor
    TP_PARTIAL_ENABLED         = False    # 1/3 partial scaling at TP1/TP2/TP3
    TP1_POSTERIOR_CEIL         = 0.93

    # ── Adaptive TP1 sizing (Option B) ─────────────────────────────────────
    # TP1 sell fraction increases when (a) near strike, (b) late in window,
    # or (c) signal degraded vs entry (posterior/edge drop).
    TP1_PARTIAL_BASE           = 0.333
    TP1_PARTIAL_MID            = 0.666
    TP1_PARTIAL_MAX            = 1.0
    TP1_CLOSE_DIST             = 80.0     # underlying points from strike considered "close"
    TP1_LATE_MIN_REM           = 3.0      # minutes remaining considered "late" for TP1 sizing
    TP1_POST_DROP_THRESH       = 0.05     # posterior drop (pp) from entry to flag degradation
    TP1_EDGE_DROP_THRESH       = 0.005    # edge drop from entry to flag degradation
    TP1_USE_EDGE_DEGRADATION   = True

    # ── Post-TP1 remainder protection (tighter trailing) ───────────────────
    # Once TP1 has filled, apply a tighter price-based trailing stop using the
    # same MFE tracking, to prevent winner→hard-stop reversals on the runner.
    TP1_TRAIL_PRICE_ACTIVATION_PCT = 0.02
    TP1_TRAIL_PRICE_DISTANCE_PCT   = 0.06

    # ── Post-TP1 runner profit protection (posterior collapse) ─────────────
    # If TP1 hit and profit exists, but posterior collapses meaningfully from
    # entry, exit the remainder.
    RUNNER_POSTERIOR_DROP_PCT       = 0.10
    RUNNER_MIN_PROFIT_PCT           = 0.03

    # ── Volatility-Adapted Stop-Loss (ATR-normalized) ────────────────────
    VOL_STOP_BASE_PCT          = 0.15     # base max drawdown
    VOL_STOP_ATR_BASELINE      = 150.0    # daily average baseline BTC ATR ($)
    VOL_STOP_MAX_PCT           = 0.30     # absolute cap on widened drawdown

    # ── Spread-Aware Exiting ─────────────────────────────────────────────
    SPREAD_AGGRESSIVE_THRESH   = 0.10     # spread >10% = don't cross aggressively
    SPREAD_EXPIRY_OVERRIDE_SEC = 60       # unless <60s to expiry
    SPREAD_MAKER_TICK          = 0.01     # place limit at best_bid + tick

    # ── Probability-Convergence Exit ─────────────────────────────────────
    PROB_CONVERGENCE_ENABLED   = True     # exit when bid >= posterior

    # ── Structural Model Reversal ────────────────────────────────────────
    MODEL_REVERSAL_DROP_PCT    = 0.15     # exit if posterior drops 15pp from entry
    MODEL_REVERSAL_MONSTER_SUPPRESS_MIN_REM = 5.0  # suppress MODEL_REVERSAL for monster entries with < 5 min remaining

    # ── Exponential Time-Decay Multiplier ────────────────────────────────
    TIME_DECAY_WINDOW_MIN      = 2.0      # activate in final 2 minutes
    TIME_DECAY_EXP_BASE        = 3.0      # exponential base for sensitivity scaling

    # Microstructure confirmation exits
    EXIT_CVD_VEL_REV_THRESH    = 0.50     # reverse CVD velocity threshold (units: from cvd_ws slope)
    EXIT_DEEP_OFI_REV_THRESH   = 0.20     # reverse normalized deep OFI threshold

    # Explicit adverse selection / book flip
    BOOK_FLIP_IMB_THRESH       = 0.18     # OBI magnitude threshold to consider a flip meaningful

    # Distribution shift / robustness gating (WILDS-inspired)
    # If model posterior diverges too far from market-implied probability, treat as shift.
    MODEL_MKT_DIVERGENCE_MAX   = 0.18

    # ── Risk / sizing (FIX #8) ────────────────────────────────────────────────
    RISK_TIER_50               = 0.15     # ≤ $50   (was 0.25 — capped at ~50% exposure)
    RISK_TIER_100              = 0.15     # $50–100
    RISK_TIER_200              = 0.12     # $100–200
    RISK_TIER_OVER             = 0.10     # > $200
    STREAK_HALVE               = True     # halve size after 2 consecutive losses
    MIN_TRADE_USD              = 5.75    # Polymarket CLOB minimum is ~$5
    MAX_TRADES_PER_WINDOW      = 5
    STREAK_HALT_AT             = 3        # halt trading after N consecutive losses
    SESSION_DRAWDOWN_HALT_PCT  = 0.30     # halt trading when session drawdown exceeds this fraction
    MICRO_EXIT_GRACE_SEC       = 90       # suppress Layer-6 micro exits for this many seconds post-entry

    # Dynamic Kelly scaling based on belief volatility (sigma_b)
    KELLY_MULT_MIN             = 0.50
    KELLY_MULT_MAX             = 1.50

    # ── BB Squeeze gate thresholds (regime-adaptive) ──────────────────────────
    BB_SQUEEZE_LOW             = 0.0015   # low-vol regime (ATR < 80)
    BB_SQUEEZE_NORMAL          = 0.0030   # normal regime
    BB_SQUEEZE_HIGH            = 0.0050   # high-vol regime (ATR > 200)

    # ── Hard capital protections ──────────────────────────────────────────────
    MAX_TRADE_USD              = 50.0     # absolute max per single trade
    MAX_TRADES_PER_HOUR        = 14        # hourly trade limit
    MAX_DAILY_TRADES           = 20        # absolute daily trade limit (reset midnight UTC)
    DAILY_LOSS_LIMIT_USD       = 12.0     # stop if daily realized loss exceeds this
    DAILY_LOSS_LIMIT_PCT       = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.15") or 0.15)

    # FOK status endpoints can be eventually consistent; re-check briefly before declaring killed.
    FOK_RECHECK_DELAY_SEC   = float(os.getenv("FOK_RECHECK_DELAY_SEC", "0.35"))

    # Profit-taking dynamic thresholds (posterior-gated, trailing guard suppressed)
    TAKE_PROFIT_STRONG_PCT = 0.20      # 25% for strong signals
    TAKE_PROFIT_STRONG_POSTERIOR = 0.90
    TAKE_PROFIT_STRONG_SCORE = 6.0
    TAKE_PROFIT_STRONG_MAX_MIN = 7.5   # Early window

    TAKE_PROFIT_MODERATE_PCT = 0.10    # 10% for moderate
    TAKE_PROFIT_MODERATE_POSTERIOR = 0.75
    TAKE_PROFIT_MODERATE_SCORE = 3.0
    TAKE_PROFIT_MODERATE_MAX_MIN = 10.0  # Mid-window

    TAKE_PROFIT_WEAK_PCT = 0.05        # 5% for weak
    TAKE_PROFIT_WEAK_POSTERIOR = 0.60
    TAKE_PROFIT_WEAK_SCORE = 1.5
    # Late window or toxic microstructure

    TAKE_PROFIT_TOXIC_MULTIPLIER = 0.5  # Reduce threshold by 50% in toxic flow
    TAKE_PROFIT_LATE_MULTIPLIER = 0.5   # Tighten near expiry
    TAKE_PROFIT_OPEN_PCT = 0.25         # Profit-taking threshold for open positions (> 25% PNL)
    MAX_SESSION_DRAWDOWN_PCT = 30.0     # Halt trading at this session drawdown %
    SESSION_DRAWDOWN_RESUME_PCT = 20.0  # Resume trading below this % (hysteresis)
    LATE_WINDOW_KELLY_MULTIPLIER = 1.5  # Increase sizing in late window with high posterior
    OUTSIDE_HOURS_POSTERIOR_MIN = 0.85  # Higher threshold outside preferred hours
    MAX_EXPOSURE_USD           = 100.0    # total notional across all positions
    KILL_SWITCH                = os.getenv("KILL_SWITCH", "false").lower() == "true"
    KILL_SWITCH_PASSWORD       = os.getenv("KILL_SWITCH_PASSWORD", "admin")

    # ── Execution modes ───────────────────────────────────────────────────────
    # When enabled, the engine records hypothetical trades in state/DB but does
    # not hit Polymarket trading endpoints. Safe for testing gates and sizing.
    PAPER_TRADE_ENABLED        = os.getenv("PAPER_TRADE_ENABLED", "false").lower() == "true"

    # ── Strike resolution priority (FIX #2) ──────────────────────────────────
    # 1. Binance 15m kline open
    # 2. Coinbase 15m kline open (NEW fallback)
    # 3. Binance mid (real spread)
    # 4. NEVER: live EMA / current price

    # ── Polymarket ────────────────────────────────────────────────────────────
    CHAIN_ID                   = 137
    POLYMARKET_HOST            = "https://clob.polymarket.com"

    POLYMARKET_PRIVATE_KEY     = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    FUNDER_ADDRESS             = os.getenv("FUNDER_ADDRESS", "")

    # ── Polygon RPC / Tokens ──────────────────────────────────────────────────
    # Switch to a dedicated Alchemy or QuickNode endpoint for better stability
    POLYGON_RPC_URL            = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com/")
    # Polymarket uses USDC.e (bridged) on Polygon — NOT native USDC
    # USDC.e: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174 (6 decimals)
    # Native: 0x3c499c542cef5e3811e1192ce70d8cc03d5c3359 (6 decimals)
    POLYGON_USDC_ADDRESS       = os.getenv("POLYGON_USDC_ADDRESS", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    POLYGON_USDC_NATIVE        = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"

    # ── API Keys ─────────────────────────────────────────────────────────────
    BINANCE_API_KEY            = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET         = os.getenv("BINANCE_API_SECRET", "")
    COINBASE_API_KEY           = os.getenv("COINBASE_API_KEY", "")
    COINBASE_API_SECRET        = os.getenv("COINBASE_API_SECRET", "")
    POLYMARKET_API_KEY         = os.getenv("POLYMARKET_API_KEY", "")
    POLYMARKET_API_SECRET      = os.getenv("POLYMARKET_API_SECRET", "")
    POLYMARKET_API_PASSPHRASE  = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    TELEGRAM_TOKEN             = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID           = os.getenv("TELEGRAM_CHAT_ID", "")
    ANTHROPIC_API_KEY          = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY              = os.getenv("GEMINI_API_KEY", "")
    DATABASE_URL               = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./state.db")
    LOG_LEVEL                  = os.getenv("LOG_LEVEL", "INFO")

    # Preferred trading times
    PREFERRED_START_HOUR_ET    = 7.0   # 7 AM ET
    PREFERRED_END_HOUR_ET      = 16.5  # 4:30 PM ET
    PREFERRED_WEEKDAYS_ONLY    = True  # Block weekends entirely

    # ── Derived helpers ───────────────────────────────────────────────────────
    @classmethod
    def is_preferred_trading_time(cls) -> bool:
        """Check if current time is within preferred trading hours (7:00 AM - 4:30 PM ET on weekdays)."""
        now_et = datetime.now(ZoneInfo("America/New_York"))
        # Get hour as a float (e.g. 16:30 -> 16.5)
        et_time_float = now_et.hour + now_et.minute / 60.0
        weekday = now_et.weekday() < 5  # Monday-Friday

        start = float(cls.PREFERRED_START_HOUR_ET)
        end = float(cls.PREFERRED_END_HOUR_ET)

        if start == end:
            in_time = True
        elif start < end:
            in_time = start <= et_time_float < end
        else:
            # Wraps past midnight
            in_time = (et_time_float >= start) or (et_time_float < end)

        return in_time and (not cls.PREFERRED_WEEKDAYS_ONLY or weekday)
    @classmethod
    def get_regime_thresholds(cls, atr14: float, balance: float = None) -> tuple[float, float]:
        """Returns (required_edge, min_score) for the given ATR regime.

        When balance < LOW_BALANCE_THRESHOLD_USD, applies a relaxed edge requirement
        so the bot can still find opportunities at low capital.
        """
        if atr14 is None:
            edge = cls.REQUIRED_EDGE_NORMAL
            score = cls.MIN_SCORE_NORMAL
        elif atr14 < cls.ATR_LOW_THRESHOLD:
            edge = cls.REQUIRED_EDGE_LOW
            score = cls.MIN_SCORE_LOW_VOL
        elif atr14 > cls.ATR_HIGH_THRESHOLD:
            edge = cls.REQUIRED_EDGE_HIGH
            score = cls.MIN_SCORE_HIGH_VOL
        else:
            edge = cls.REQUIRED_EDGE_NORMAL
            score = cls.MIN_SCORE_NORMAL

        # Balance-adaptive override: relax edge requirement at low capital
        if balance is not None and balance < cls.LOW_BALANCE_THRESHOLD_USD:
            edge = min(edge, cls.REQUIRED_EDGE_LOW_BALANCE)

        return edge, score

    @classmethod
    def get_risk_pct(cls, balance: float) -> float:
        """Returns risk fraction for the given balance tier."""
        if balance <= 50:
            return cls.RISK_TIER_50
        if balance <= 100:
            return cls.RISK_TIER_100
        if balance <= 200:
            return cls.RISK_TIER_200
        return cls.RISK_TIER_OVER

    @classmethod
    def validate(cls):
        missing = []
        if not cls.POLYMARKET_PRIVATE_KEY:
            missing.append("POLYMARKET_PRIVATE_KEY")
        if not cls.FUNDER_ADDRESS:
            pass
        if missing:
            raise ValueError(f"CRITICAL: Missing required environment variables: {', '.join(missing)}")

# Fail fast at module load time if running the bot
Config.validate()
