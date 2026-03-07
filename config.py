import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    # ── Timing ────────────────────────────────────────────────────────────────
    LOOP_INTERVAL_SEC          = 15       # inner loop cadence
    WINDOW_SEC                 = 900      # 15-minute binary window
    EARLY_WINDOW_GUARD_MIN     = 6.0      # block non-monster trades in first 6 min
    BELIEF_VOL_LOOKBACK_SEC    = 180      # rolling σ_B window (3 min)
    BELIEF_VOL_DEFAULT         = 0.15
    BELIEF_VOL_MULT_MIN        = 0.70
    BELIEF_VOL_MULT_MAX        = 2.00
    BELIEF_VOL_LATE_MAX        = 1.50     # cap when < 5 min remain

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
    REQUIRED_EDGE_NORMAL       = 0.035
    REQUIRED_EDGE_HIGH         = 0.025
    DEFAULT_ATR                = 150.0    # Fallback for Bayesian posterior if TAAPI is 429

    # ── Late-window conviction override ──────────────────────────────────────
    # When near expiry with very high posterior and large distance-from-strike,
    # the edge gate is relaxed because the market already prices in the outcome.
    LATE_CONVICTION_MIN_REM    = 3.0      # must be within last 3 min of window
    LATE_CONVICTION_POSTERIOR  = 0.93     # model must be ≥93% confident
    LATE_CONVICTION_DISTANCE   = 50.0     # price must be ≥$50 from strike
    LATE_CONVICTION_EDGE       = 0.025    # relaxed edge requirement (2.5% instead of 3.5%)

    # ── ADX trend filter (FIX #5) ─────────────────────────────────────────────
    ADX_TREND_THRESHOLD        = 20.0     # below = choppy, block directional entry

    # ── OFI / microstructure ─────────────────────────────────────────────────
    OFI_STRONG                 = 5.0
    OFI_15M                    = 15.0
    VPIN_BLOCK_THRESHOLD       = 0.70     # block entry if vpin > this AND score weak

    # ── Scoring constants ─────────────────────────────────────────────────────
    BLIND_ENTRY_SCORE          = 7.0      # legacy; use min_score_* in practice
    MONSTER_SCORE              = 8.0
    MONSTER_POSTERIOR          = 0.90
    STOP_LOSS_DELTA            = 4.0

    # ── Exit parameters ───────────────────────────────────────────────────────
    TAKE_PROFIT_PRICE          = 0.97
    MAX_DRAWDOWN_PCT           = 0.08
    FORCED_LATE_EXIT_MIN_REM   = 5.0
    FORCED_DISTANCE_EXIT_MIN_REM = 3.0
    FORCED_PROFIT_LOCK_MIN_REM = 2.0
    FORCED_LATE_LOSS_PCT       = 0.08
    FORCED_PROFIT_PCT          = 0.25
    FORCED_DISTANCE_MAX        = 30.0     # abs(btcPrice - strike) < this triggers late exit

    # ── Risk / sizing (FIX #8) ────────────────────────────────────────────────
    RISK_TIER_50               = 0.25     # ≤ $50   (was 1.0 — the ruin bug)
    RISK_TIER_100              = 0.25     # $50–100
    RISK_TIER_200              = 0.20     # $100–200
    RISK_TIER_OVER             = 0.15     # > $200
    STREAK_HALVE               = True     # halve size after 2 consecutive losses
    MIN_TRADE_USD              = 5.75
    MAX_TRADES_PER_WINDOW      = 3
    STREAK_HALT_AT             = 3        # halt trading after N consecutive losses

    # ── Hard capital protections ──────────────────────────────────────────────
    MAX_TRADE_USD              = 50.0     # absolute max per single trade
    MAX_TRADES_PER_HOUR        = 8        # hourly trade limit
    DAILY_LOSS_LIMIT_USD       = 25.0     # stop if daily realized loss exceeds this
    DAILY_LOSS_LIMIT_PCT       = 0.10     # stop if daily loss > 10% of starting balance
    MAX_EXPOSURE_USD           = 100.0    # total notional across all positions
    KILL_SWITCH                = os.getenv("KILL_SWITCH", "false").lower() == "true"
    KILL_SWITCH_PASSWORD       = os.getenv("KILL_SWITCH_PASSWORD", "admin")

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
    DATABASE_URL               = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./state.db")
    LOG_LEVEL                  = os.getenv("LOG_LEVEL", "INFO")

    # ── Derived helpers ───────────────────────────────────────────────────────
    @classmethod
    def get_regime_thresholds(cls, atr14: float) -> tuple[float, float]:
        """Returns (required_edge, min_score) for the given ATR regime."""
        if atr14 is None:
            return cls.REQUIRED_EDGE_NORMAL, cls.MIN_SCORE_NORMAL
        if atr14 < cls.ATR_LOW_THRESHOLD:
            return cls.REQUIRED_EDGE_LOW, cls.MIN_SCORE_LOW_VOL
        if atr14 > cls.ATR_HIGH_THRESHOLD:
            return cls.REQUIRED_EDGE_HIGH, cls.MIN_SCORE_HIGH_VOL
        return cls.REQUIRED_EDGE_NORMAL, cls.MIN_SCORE_NORMAL

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
