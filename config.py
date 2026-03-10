import os
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()


class Config:
    # ── Timing ────────────────────────────────────────────────────────────────
    LOOP_INTERVAL_SEC          = 3        # inner loop cadence (3s: faster reversal detection; well within Binance rate limits)
    WINDOW_SEC                 = 900      # 15-minute binary window
    EARLY_WINDOW_GUARD_MIN     = 4.0      # block non-monster trades in first 4 min (entries only in last 7.5 min)
    BELIEF_VOL_LOOKBACK_SEC    = 180      # rolling σ_B window (3 min)
    BELIEF_VOL_DEFAULT         = 0.15
    BELIEF_VOL_MULT_MIN        = 0.70
    # Belief-volatility multiplier caps: keep regression toward 0.5 gentle, not crushing.
    BELIEF_VOL_MULT_MAX        = 1.30     # normal cap when more than 5 min remain
    BELIEF_VOL_LATE_MAX        = 1.15     # tighter cap when < 5 min remain
    # Optional: scale belief-volatility effect by time remaining in the window.
    # When enabled, high σ_B early in the candle has more influence than very late.
    BELIEF_VOL_TIME_DECAY_ENABLED = False

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
    LOW_BALANCE_THRESHOLD_USD  = 20.0   # apply lower edge when balance < this
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

    # ── ADX trend filter (FIX #5) ─────────────────────────────────────────────
    ADX_TREND_THRESHOLD        = 20.0     # below = choppy, block directional entry

    # ── OFI / microstructure ─────────────────────────────────────────────────
    OFI_STRONG                 = 5.0
    OFI_15M                    = 15.0
    VPIN_BLOCK_THRESHOLD       = 0.85     # block entry if vpin > this AND score weak (raised: fewer false halts)

    # ── VPIN toxicity regimes (exit/hold shaping) ─────────────────────────────
    VPIN_TOXIC_THRESHOLD       = 0.85     # VPIN above this = toxic flow regime
    VPIN_TOXIC_HOLD_MAX_SEC    = 240.0    # shorten holding in toxic regime (theta + adverse selection)

    # ── Scoring constants ─────────────────────────────────────────────────────
    BLIND_ENTRY_SCORE          = 7.0      # legacy; use min_score_* in practice
    MONSTER_SCORE              = 8.0
    MONSTER_POSTERIOR          = 0.90
    STOP_LOSS_DELTA            = 7.0

    # ── Exit parameters ───────────────────────────────────────────────────────
    TAKE_PROFIT_PRICE          = 0.99
    SLIPPAGE_BUFFER_PCT        = 0.008    # Phase 6: 80bps execution buffer
    MAX_DRAWDOWN_PCT           = 0.08    # Tightened from 0.12 — trigger FORCED_DRAWDOWN earlier
    HARD_STOP_PCT              = 0.25    # HARD unconditional circuit breaker — no posterior gate.
                                          # Trailing guard held a -65% loss when posterior lagged price.
                                          # At -25% the position is unrecoverable on a 15m binary; cut always.
    FORCED_LATE_EXIT_MIN_REM    = 1.0     # Force exit at <1 min if losing
    FORCED_DISTANCE_EXIT_MIN_REM = 3.0
    FORCED_PROFIT_LOCK_MIN_REM = 2.0
    FORCED_LATE_LOSS_PCT       = 0.10   # tightened from 0.15 — cut losers faster in last 5 min
    FORCED_PROFIT_PCT          = 0.25
    FORCED_DISTANCE_MAX        = 30.0     # abs(btcPrice - strike) < this triggers late exit

    # Volatility-adjusted trailing stop (probability-space, ATR-scaled)
    TRAIL_ATR_REF              = 120.0    # reference ATR level for scaling trailing width
    TRAIL_BASE_POST_DROP       = 0.04     # base allowable posterior drop from peak before exit
    TRAIL_ATR_SCALE            = 0.06     # additional allowed drop at high ATR (scaled by atr/ref)
    TRAIL_MIN_POST_DROP        = 0.02     # floor (tightest) trailing width
    TRAIL_MAX_POST_DROP        = 0.12     # cap (widest) trailing width
    TRAIL_ARM_MIN_PROFIT_PCT   = 0.01     # only arm trailing after at least +1% unrealized
    TRAIL_MIN_HOLD_SEC         = 30.0     # minimum hold before trailing can trigger

    # Microstructure confirmation exits
    EXIT_CVD_VEL_REV_THRESH    = 0.50     # reverse CVD velocity threshold (units: from cvd_ws slope)
    EXIT_DEEP_OFI_REV_THRESH   = 0.20     # reverse normalized deep OFI threshold

    # Explicit adverse selection / book flip
    BOOK_FLIP_IMB_THRESH       = 0.18     # OBI magnitude threshold to consider a flip meaningful
    BOOK_FLIP_CONFIRM_CYCLES   = 2        # require N consecutive flips to trigger

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
    STREAK_HALT_AT             = 5        # halt trading after N consecutive losses

    # Dynamic Kelly scaling based on belief volatility (sigma_b)
    KELLY_MULT_MIN             = 0.50
    KELLY_MULT_MAX             = 1.50

    # ── BB Squeeze gate thresholds (regime-adaptive) ──────────────────────────
    BB_SQUEEZE_LOW             = 0.0015   # low-vol regime (ATR < 80)
    BB_SQUEEZE_NORMAL          = 0.0030   # normal regime
    BB_SQUEEZE_HIGH            = 0.0050   # high-vol regime (ATR > 200)

    # ── Hard capital protections ──────────────────────────────────────────────
    MAX_TRADE_USD              = 50.0     # absolute max per single trade
    MAX_TRADES_PER_HOUR        = 8        # hourly trade limit
    DAILY_LOSS_LIMIT_USD       = 25.0     # stop if daily realized loss exceeds this
    DAILY_LOSS_LIMIT_PCT       = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.15") or 0.15)

    # Profit-taking dynamic thresholds (posterior-gated, trailing guard suppressed)
    TAKE_PROFIT_STRONG_PCT = 0.25      # 25% for strong signals
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
    PREFERRED_HOURS_UTC = [(11.0, 22.0)]  # 7 AM - 6 PM ET on weekdays
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
    CHAIN_ID                   = 137      # Polygon mainnet (use 80002 for Amoy testnet)
    POLYMARKET_HOST            = "https://clob.polymarket.com"

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
    POLYMARKET_PRIVATE_KEY     = os.getenv("POLYMARKET_PRIVATE_KEY", "")
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
    PREFERRED_START_HOUR_ET    = 6   # 6 AM ET
    PREFERRED_END_HOUR_ET      = 18  # 6 PM ET
    PREFERRED_WEEKDAYS_ONLY    = True

    # ── Derived helpers ───────────────────────────────────────────────────────
    @classmethod
    def is_preferred_trading_time(cls) -> bool:
        """Check if current time is within preferred trading hours (6am-6pm ET on weekdays)."""
        now_utc = datetime.now(timezone.utc)
        # Approximate ET as UTC-4 (Eastern Time, ignoring DST for simplicity)
        et_hour = (now_utc.hour - 4) % 24
        weekday = now_utc.weekday() < 5  # Monday-Friday
        in_time = cls.PREFERRED_START_HOUR_ET <= et_hour < cls.PREFERRED_END_HOUR_ET
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
        # Require Polymarket credentials to run
        for key in ["POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY", "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE"]:
            if not getattr(cls, key):
                missing.append(key)
        if missing:
            raise ValueError(f"CRITICAL: Missing required environment variables: {', '.join(missing)}")

# Fail fast at module load time if running the bot
Config.validate()
