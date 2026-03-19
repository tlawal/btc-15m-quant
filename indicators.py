"""
Indicator fetching and computation (locally).
All indicators are computed locally from Binance klines.
"""

import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from data_feeds import Candle

log = logging.getLogger(__name__)

@dataclass
class Indicators:
    # Technical Indicators
    ema9:      Optional[float] = None
    ema20:     Optional[float] = None
    rsi14:     Optional[float] = None
    atr14:     Optional[float] = None
    macd_hist: Optional[float] = None
    mfi14:     Optional[float] = None
    obv:       Optional[float] = None
    vwma15:    Optional[float] = None
    adx14:     Optional[float] = None
    stoch_k:   Optional[float] = None
    stoch_d:   Optional[float] = None
    bb_upper:  Optional[float] = None
    bb_mid:    Optional[float] = None
    bb_lower:  Optional[float] = None
    bb_width:  Optional[float] = None

    # Computed slopes
    obv_slope:  Optional[float] = None
    price_slope: Optional[float] = None

    # Raw current values
    close:   Optional[float] = None
    volume:  Optional[float] = None

def compute_local_indicators(
    klines_5m: List[Candle],
    klines_1m: List[Candle],
) -> Indicators:
    """
    Compute all technical indicators locally using pandas.
    """
    if not klines_5m or not klines_1m:
        return Indicators()

    # --- Convert to DataFrames ---
    def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
        df = pd.DataFrame([{
            "ts": c.ts_ms,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        } for c in candles])
        df.set_index("ts", inplace=True)
        return df

    df5 = candles_to_df(klines_5m)
    df1 = candles_to_df(klines_1m)

    if len(df5) < 30: # Minimum history for some indicators
        log.warning(f"Insufficient kline history: {len(df5)}")
        return Indicators()

    res = Indicators()
    res.close = df5['close'].iloc[-1]
    res.volume = df5['volume'].iloc[-1]

    # --- 5m Indicators ---
    # EMA
    res.ema9 = df5['close'].ewm(span=9, adjust=False).mean().iloc[-1]
    res.ema20 = df5['close'].ewm(span=20, adjust=False).mean().iloc[-1]

    # RSI — Wilder's exponential smoothing (alpha=1/14)
    delta = df5['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    res.rsi14 = 100 - (100 / (1 + rs)).iloc[-1]

    # ATR — Wilder's exponential smoothing
    high_low = df5['high'] - df5['low']
    high_pc = (df5['high'] - df5['close'].shift()).abs()
    low_pc = (df5['low'] - df5['close'].shift()).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    res.atr14 = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    # MACD
    exp1 = df5['close'].ewm(span=12, adjust=False).mean()
    exp2 = df5['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    res.macd_hist = (macd - signal).iloc[-1]

    # MFI
    typical_price = (df5['high'] + df5['low'] + df5['close']) / 3
    raw_money_flow = typical_price * df5['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0)
    m_ratio = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum().replace(0, float('nan'))
    mfi_val = 100 - (100 / (1 + m_ratio)).iloc[-1]
    res.mfi14 = mfi_val if pd.notna(mfi_val) and not np.isinf(mfi_val) else None

    # ADX — corrected directional movement with Wilder's smoothing
    up_move = df5['high'].diff()
    down_move = -df5['low'].diff()  # low[i-1] - low[i]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
    res.adx14 = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    # Stochastic
    low_min = df5['low'].rolling(window=14).min()
    high_max = df5['high'].rolling(window=14).max()
    k = 100 * (df5['close'] - low_min) / (high_max - low_min)
    res.stoch_k = k.iloc[-1]
    res.stoch_d = k.rolling(window=3).mean().iloc[-1]

    # Bollinger Bands
    res.bb_mid = df5['close'].rolling(window=20).mean().iloc[-1]
    std = df5['close'].rolling(window=20).std().iloc[-1]
    res.bb_upper = res.bb_mid + (std * 2)
    res.bb_lower = res.bb_mid - (std * 2)
    if abs(res.bb_mid) > 1e-9:
        res.bb_width = (res.bb_upper - res.bb_lower) / res.bb_mid

    # --- 1m Indicators ---
    # OBV
    obv = (np.sign(df1['close'].diff()) * df1['volume']).fillna(0).cumsum()
    res.obv = obv.iloc[-1]

    # VWMA (1m)
    res.vwma15 = ( (df1['close'] * df1['volume']).rolling(window=15).sum() / df1['volume'].rolling(window=15).sum() ).iloc[-1]

    # --- Slopes ---
    # OBV Slope (last 10 mins — 5 was too noisy for a ±2.5 weighted signal)
    if len(obv) >= 10:
        res.obv_slope = obv.iloc[-1] - obv.iloc[-10]
    elif len(obv) >= 5:
        res.obv_slope = obv.iloc[-1] - obv.iloc[-5]

    # Price Slope — use 1m data (same timeframe as OBV) for apples-to-apples divergence
    if len(df1) >= 10:
        res.price_slope = df1['close'].iloc[-1] - df1['close'].iloc[-10]
    elif len(df1) >= 5:
        res.price_slope = df1['close'].iloc[-1] - df1['close'].iloc[-5]

    return res

# ── Math helpers ─────────────────────────────────────────────────────────────

def normal_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    abs_x = abs(x) / math.sqrt(2)
    t = 1 / (1 + p * abs_x)
    y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-abs_x * abs_x)
    return 0.5 * (1 + sign * y)

def student_t_cdf(x: float, df: int = 6) -> float:
    """Student-t CDF with heavier tails than Gaussian. df=6 fits BTC returns.

    Not currently wired into the posterior pipeline — available for activation
    after calibration data confirms Gaussian overconfidence at tail z-scores.
    """
    try:
        from scipy.stats import t as t_dist
        return float(t_dist.cdf(x, df))
    except ImportError:
        # Fallback: for moderate df and typical z-scores, approximate via
        # scaled normal CDF with wider tails
        scale = math.sqrt(df / (df - 2)) if df > 2 else 1.5
        return normal_cdf(x / scale)

def logit(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1 - p))

def inv_logit(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── Fix #6: Higher-timeframe trend detection ─────────────────────────────────

def compute_htf_trend(klines_1h) -> str:
    """Compute 1H trend direction from hourly klines using EMA9/EMA20 crossover.

    Args:
        klines_1h: List of Candle objects (or any object with .close attribute).
                   Needs at least 21 candles.
    Returns:
        "UP", "DOWN", or "NEUTRAL"
    """
    if not klines_1h or len(klines_1h) < 21:
        return "NEUTRAL"

    closes = [c.close for c in klines_1h]

    def _ema(data, period):
        alpha = 2.0 / (period + 1)
        val = data[0]
        for px in data[1:]:
            val = alpha * px + (1 - alpha) * val
        return val

    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)

    # Require meaningful separation (0.05% of price) to avoid noise
    threshold = closes[-1] * 0.0005 if closes[-1] > 0 else 1.0
    diff = ema9 - ema20

    if diff > threshold:
        return "UP"
    elif diff < -threshold:
        return "DOWN"
    return "NEUTRAL"


# ── Fix #10: Candlestick pattern recognition ─────────────────────────────────

def compute_candle_patterns(klines) -> dict:
    """Detect common candlestick patterns from the last 2 candles.

    Args:
        klines: List of Candle objects (needs at least 2).
    Returns:
        Dict with boolean flags: bearish_engulfing, bullish_engulfing,
        doji, shooting_star, hammer.
    """
    result = {
        "bearish_engulfing": False,
        "bullish_engulfing": False,
        "doji": False,
        "shooting_star": False,
        "hammer": False,
    }

    if not klines or len(klines) < 2:
        return result

    prev = klines[-2]
    curr = klines[-1]

    curr_body = abs(curr.close - curr.open)
    curr_range = curr.high - curr.low
    prev_body = abs(prev.close - prev.open)

    if curr_range <= 0:
        return result

    # Doji: body <= 15% of range
    if curr_body <= 0.15 * curr_range:
        result["doji"] = True

    # Bearish engulfing: prev was green, current red engulfs it
    if (prev.close > prev.open                     # prev is green
            and curr.open >= prev.close             # current opens at/above prev close
            and curr.close <= prev.open             # current closes at/below prev open
            and curr_body > prev_body):             # current body is larger
        result["bearish_engulfing"] = True

    # Bullish engulfing: prev was red, current green engulfs it
    if (prev.close < prev.open                     # prev is red
            and curr.open <= prev.close             # current opens at/below prev close
            and curr.close >= prev.open             # current closes at/above prev open
            and curr_body > prev_body):             # current body is larger
        result["bullish_engulfing"] = True

    # Shooting star: upper shadow >= 2x body, small lower shadow, bearish
    upper_shadow = curr.high - max(curr.open, curr.close)
    lower_shadow = min(curr.open, curr.close) - curr.low
    if (upper_shadow >= 2 * curr_body
            and lower_shadow <= 0.3 * curr_range
            and curr.close < curr.open):            # bearish close
        result["shooting_star"] = True

    # Hammer: lower shadow >= 2x body, small upper shadow, bullish
    if (lower_shadow >= 2 * curr_body
            and upper_shadow <= 0.3 * curr_range
            and curr.close > curr.open):            # bullish close
        result["hammer"] = True

    return result

