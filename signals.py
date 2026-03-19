"""
Feature calculation, score weighting, and Bayesian momentum signals.
Extracted from logic.py during Phase 3 refactor.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from config import Config
from indicators import (Indicators, normal_cdf, student_t_cdf, logit, inv_logit, clamp,
                         compute_htf_trend, compute_candle_patterns)
from state import EngineState, BeliefVolSample

log = logging.getLogger(__name__)

@dataclass
class SignalResult:
    # Regime
    regime:               str    = "normal"     # "low" | "normal" | "high"
    atr14:                Optional[float] = None
    expected_move:        Optional[float] = None

    # Strike / distance
    strike_price:         Optional[float] = None
    strike_source:        str    = "none"
    distance:             Optional[float] = None
    z_score:              Optional[float] = None

    # Bayesian posteriors
    posterior_fair_up:    Optional[float] = None
    posterior_fair_down:  Optional[float] = None
    posterior_final_up:   Optional[float] = None
    posterior_final_down: Optional[float] = None
    # Optional, score-adjusted posteriors (for analysis/experiments)
    posterior_adj_up:     Optional[float] = None
    posterior_adj_down:   Optional[float] = None

    # Belief volatility
    sigma_b:              float  = 0.15
    bvol_multiplier:      float  = 1.0
    current_x:            Optional[float] = None

    # Scoring
    signed_score:         float  = 0.0
    abs_score:            float  = 0.0

    # direction from posterior
    direction:            str    = "NEUTRAL"    # "UP" | "DOWN" | "NEUTRAL"

    # Component scores
    ema_score:            float  = 0.0
    vwap_score:           float  = 0.0
    rsi_score:            float  = 0.0
    atr_score:            float  = 0.0
    macd_score:           float  = 0.0
    stoch_score:          float  = 0.0
    mfi_score:            float  = 0.0
    obv_score:            float  = 0.0
    cvd_score:            float  = 0.0
    ofi_score:            float  = 0.0
    imbalance_score:      float  = 0.0
    flow_accel_score:     float  = 0.0

    # Phase 3: new signal scores
    tob_score:            float  = 0.0
    cvd_velocity_score:   float  = 0.0
    pm_flow_score:        float  = 0.0
    whale_flow_score:     float  = 0.0   # Tier 1: large-fill directional score
    spread_skew_score:    float  = 0.0   # Tier 1: bid/ask spread asymmetry
    window_momentum_score:float  = 0.0   # Tier 1: cross-window trend persistence
    liq_cascade_score:    float  = 0.0   # Tier 2: Binance liquidation cascade
    funding_delta_score:  float  = 0.0   # Tier 2: funding rate change signal

    # Phase 2: new signal scores
    liq_vacuum_score:     float  = 0.0
    bb_position_score:    float  = 0.0
    cross_exch_score:     float  = 0.0
    accum_ofi_score:      float  = 0.0
    misprice_score:       float  = 0.0
    mtf_momentum_score:   float  = 0.0
    adx_stoch_boost:      float  = 0.0
    spread_pressure_score:float  = 0.0
    oracle_lag_score:     float  = 0.0
    funding_rate_score:   float  = 0.0

    # Oracle / Perps info
    oracle_px:            Optional[float] = None
    oracle_update_age:    Optional[float] = None
    funding_rate:         Optional[float] = None
    perp_basis_pct:       Optional[float] = None
    basis_edge:           Optional[float] = None

    # Deep LOB (40-level)
    lob_imbalance_40:      Optional[float] = None
    lob_imbalance_40_norm: Optional[float] = None
    lob_spread:            Optional[float] = None
    lob_vwap_dev:          Optional[float] = None
    lob_mom_change:        Optional[float] = None

    # Model ensemble
    model_prob_up:         Optional[float] = None

    # Hawkes timing proxy
    hawkes_next_event_sec: Optional[float] = None

    # Edge
    edge_up:              Optional[float] = None
    edge_down:            Optional[float] = None
    target_edge:          Optional[float] = None
    target_side:          Optional[str]   = None   # "YES" | "NO"

    # Thresholds (regime-adaptive)
    required_edge:        float  = 0.035
    min_score:            float  = 4.0

    # Volatility surface (logging-only, not applied to stops yet)
    vol_surface_adj:      Optional[float] = None

    # Skip gates
    skip_gates:           list   = field(default_factory=list)
    monster_signal:       bool   = False

    # Micro
    vpin_proxy:           float  = 0.0
    deep_imbalance:       float  = 0.5
    deep_ofi:             float  = 0.0
    cvd:                  float  = 0.0
    microprice:           Optional[float] = None
    obi:                  float  = 0.0
    bid_depth:            float  = 0.0
    ask_depth:            float  = 0.0

    # Reporting extras
    sizing:               float  = 0.0
    score_delta:          Optional[float] = None
    price_delta:          Optional[float] = None
    yes_mid:              Optional[float] = None
    no_mid:               Optional[float] = None
    model_score:          Optional[float] = None    # Phase 5: ML probability of UP

    def to_feature_dict(self) -> dict:
        """Phase 5: Convert signal results to a flat dict for ML logging."""
        d = {
            "atr14":              self.atr14,
            "expected_move":      self.expected_move,
            "z_score":            self.z_score,
            "posterior_fair_up":  self.posterior_fair_up,
            "posterior_final_up": self.posterior_final_up,
            "posterior_adj_up":   self.posterior_adj_up,
            "sigma_b":            self.sigma_b,
            "bvol_multiplier":    self.bvol_multiplier,
            "signed_score":       self.signed_score,
            "ema_score":          self.ema_score,
            "vwap_score":         self.vwap_score,
            "rsi_score":          self.rsi_score,
            "atr_score":          self.atr_score,
            "macd_score":         self.macd_score,
            "stoch_score":        self.stoch_score,
            "mfi_score":          self.mfi_score,
            "obv_score":          self.obv_score,
            "cvd_score":          self.cvd_score,
            "ofi_score":          self.ofi_score,
            "imbalance_score":    self.imbalance_score,
            "flow_accel_score":   self.flow_accel_score,
            "liq_vacuum_score":   self.liq_vacuum_score,
            "bb_position_score":  self.bb_position_score,
            "cross_exch_score":   self.cross_exch_score,
            "accum_ofi_score":    self.accum_ofi_score,
            "misprice_score":     self.misprice_score,
            "mtf_momentum_score": self.mtf_momentum_score,
            "adx_stoch_boost":    self.adx_stoch_boost,
            "spread_pressure_score": self.spread_pressure_score,
            "oracle_lag_score":   self.oracle_lag_score,
            "funding_rate_score": self.funding_rate_score,
            "funding_rate":       self.funding_rate,
            "perp_basis_pct":     self.perp_basis_pct,
            "basis_edge":         self.basis_edge,
            "lob_imbalance_40":      self.lob_imbalance_40,
            "lob_imbalance_40_norm": self.lob_imbalance_40_norm,
            "lob_spread":            self.lob_spread,
            "lob_vwap_dev":          self.lob_vwap_dev,
            "lob_mom_change":        self.lob_mom_change,
            "model_prob_up":         self.model_prob_up,
            "hawkes_next_event_sec": self.hawkes_next_event_sec,
            "tob_score":          self.tob_score,
            "cvd_velocity_score": self.cvd_velocity_score,
            "pm_flow_score":      self.pm_flow_score,
            "vpin_proxy":         self.vpin_proxy,
            "deep_imbalance":     self.deep_imbalance,
            "deep_ofi":           self.deep_ofi,
            "cvd":                self.cvd,
            "obi":                self.obi,
            "yes_mid":            self.yes_mid,
            "no_mid":             self.no_mid,
        }
        # Filter out Nones
        return {k: v for k, v in d.items() if v is not None}

def _compute_vol_surface(yes_mid: float, atr14: float, minutes_remaining: float) -> float:
    """Implied volatility surface adjustment (logging-only).

    Combines market-implied vol (from yes_mid distance to 0.50) with
    realized vol (ATR) and time-to-expiry to produce a multiplier.
    Higher values = more volatile / wider stops appropriate.

    Returns clamped [0.5, 2.0]. NOT applied to stop widths yet —
    logged on SignalResult.vol_surface_adj for validation.
    """
    if atr14 is None or atr14 <= 0 or yes_mid is None:
        return 1.0

    # Market-implied component: how far yes_mid is from 0.50
    # Near 0.50 = uncertain = higher vol; near 0/1 = resolved = lower vol
    market_uncertainty = 1.0 - 2.0 * abs(yes_mid - 0.50)  # [0, 1]

    # Time component: less time = higher urgency = higher vol surface
    time_factor = max(0.5, min(2.0, 15.0 / max(1.0, minutes_remaining)))

    # ATR component: normalize to a baseline (~$200 for BTC)
    atr_factor = atr14 / 200.0

    vol_adj = market_uncertainty * time_factor * atr_factor
    return max(0.5, min(2.0, vol_adj))


def compute_early_minute_momentum(*, klines_1m=None, window_start_ts=None):
    """STUB: Early-minute BTC momentum signal.

    Concept: Use BTC price momentum in the first 1-2 minutes of a 15m window
    to predict direction for the remainder. If BTC moves strongly in one direction
    immediately after window open, momentum tends to persist for the rest of the
    window (microstructure anticipation / opening price discovery).

    NOT IMPLEMENTED — returns None. Will require:
    - 1m klines specifically from the current window's first 2 minutes
    - Comparison of close[t=1min] vs open[t=0] as a % of ATR
    - Threshold calibration from historical data
    - Integration as a signal modifier (boost conviction when early momentum agrees)
    """
    return None


def compute_signals(
    *,
    indic:            Indicators,
    btc_price:        float,
    minutes_remaining: float,
    now_ts:           int,
    state:            EngineState,
    strike:           Optional[float],
    strike_source:    str,
    # Microstructure
    bid_depth20:      float,
    ask_depth20:      float,
    bid_levels:       Optional[Sequence] = None,
    ask_levels:       Optional[Sequence] = None,
    deep_imbalance:   float,
    vpin_proxy:       float,
    deep_ofi:         float,
    microprice:       Optional[float],
    is_stale_micro:   bool,
    # Real CVD
    cvd_delta:        float,
    true_cvd:         float,
    # Phase 2: new signal inputs
    accumulated_ofi:  float,
    cross_cvd_agree:  bool,
    cvd_total_vol:    float,
    prev_cvd_total_vol: float,
    oracle_px:        Optional[float],
    oracle_update_ts: Optional[int],
    funding_rate:     Optional[float],
    perp_basis_pct:   Optional[float] = None,
    # Phase 3: TOB + CVD velocity + Polymarket flow
    tob_imbalance:    float = 0.5,
    cvd_velocity:     float = 0.0,
    arrival_ts_ms:    Optional[Sequence[int]] = None,
    pm_net_flow:      float = 0.0,
    # Polymarket
    yes_mid:          Optional[float] = None,
    no_mid:           Optional[float] = None,
    yes_bid:          Optional[float] = None,
    yes_ask:          Optional[float] = None,
    no_bid:           Optional[float] = None,
    no_ask:           Optional[float] = None,
    total_bid_size:   float = 0.0,
    total_ask_size:   float = 0.0,
    # Tier 1 new signals
    whale_flow:       float = 0.0,   # USD net whale flow (>$50 fills)
    window_outcomes:  list  = None,  # list of recent window outcomes ["UP","DOWN",...]
    liq_cascade:      float = 0.0,   # net liquidation USD (positive = long liqs)
    funding_rate_prev:Optional[float] = None,  # funding rate from 1h ago
    # Phase 5
    inference_engine: Optional[object] = None,
    # Phase 4 Optimization
    score_offset:     float = 0.0,
    edge_offset:      float = 0.0,
    # Balance-adaptive edge
    balance:          float = None,
    # Late-window entry hardening
    one_sided_cycles: int = 0,
    # Fix #6: Higher-timeframe trend
    htf_trend:        str = "NEUTRAL",   # "UP" | "DOWN" | "NEUTRAL"
    # Fix #9: Volume Profile Point of Control
    vpoc:             Optional[float] = None,
    # Fix #10: Candlestick patterns
    candle_patterns:  Optional[dict] = None,
) -> SignalResult:

    res = SignalResult()

    window_min = Config.WINDOW_SEC / 60.0
    t_frac = minutes_remaining / window_min if window_min > 0 else 0.5

    # ── Regime ───────────────────────────────────────────────────────
    atr14 = indic.atr14
    res.atr14 = atr14
    if atr14 is None:
        res.regime = "normal"
    elif atr14 < Config.ATR_LOW_THRESHOLD:
        res.regime = "low"
    elif atr14 > Config.ATR_HIGH_THRESHOLD:
        res.regime = "high"
    else:
        res.regime = "normal"

    res.required_edge, res.min_score = Config.get_regime_thresholds(atr14, balance=balance)
    res.required_edge += edge_offset
    res.min_score += score_offset


    # ── Expected move ─────────────────────────────────────────────────────────
    res.expected_move = 0.0
    effective_atr = atr14 or Config.DEFAULT_ATR
    if minutes_remaining > 0:
        res.expected_move = effective_atr * math.sqrt(minutes_remaining / 5.0)
    res.strike_price   = strike
    res.strike_source  = strike_source
    res.yes_mid        = yes_mid
    res.no_mid         = no_mid

    res.perp_basis_pct = perp_basis_pct
    if perp_basis_pct is not None:
        try:
            res.basis_edge = abs(float(perp_basis_pct))
        except Exception:
            res.basis_edge = None

    # ── Deep LOB features (40-level) ─────────────────────────────────────────
    def _compute_deep_lob_feats(
        bids: Optional[Sequence],
        asks: Optional[Sequence],
        levels: int = 40,
    ) -> dict:
        if not bids or not asks:
            return {}
        try:
            bb = []
            aa = []
            for px, sz in list(bids)[:levels]:
                bb.append((float(px), float(sz)))
            for px, sz in list(asks)[:levels]:
                aa.append((float(px), float(sz)))
        except Exception:
            return {}

        if not bb or not aa:
            return {}

        bid_vol = sum(sz for _, sz in bb)
        ask_vol = sum(sz for _, sz in aa)
        imb = bid_vol - ask_vol
        denom = bid_vol + ask_vol
        imb_norm = (imb / denom) if denom > 0 else 0.0

        best_bid = bb[0][0] if bb else None
        best_ask = aa[0][0] if aa else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        num = sum(px * sz for px, sz in bb) + sum(px * sz for px, sz in aa)
        den = bid_vol + ask_vol
        depth_vwap = (num / den) if den > 0 else None
        mid = ((best_bid + best_ask) / 2.0) if (best_bid and best_ask) else None
        vwap_dev = (depth_vwap - mid) if (depth_vwap is not None and mid is not None) else None

        # Momentum-change proxy: difference between 40-level and 10-level imbalance.
        # This captures whether pressure is strengthening deeper in the book.
        b10 = bb[:10]
        a10 = aa[:10]
        bid10 = sum(sz for _, sz in b10)
        ask10 = sum(sz for _, sz in a10)
        imb10 = bid10 - ask10
        denom10 = bid10 + ask10
        imb10_norm = (imb10 / denom10) if denom10 > 0 else 0.0
        mom_change = imb_norm - imb10_norm

        return {
            "lob_imbalance_40": imb,
            "lob_imbalance_40_norm": imb_norm,
            "lob_spread": spread,
            "lob_vwap_dev": vwap_dev,
            "lob_mom_change": mom_change,
        }

    deep_feats = _compute_deep_lob_feats(bid_levels, ask_levels, levels=40)
    res.lob_imbalance_40 = deep_feats.get("lob_imbalance_40")
    res.lob_imbalance_40_norm = deep_feats.get("lob_imbalance_40_norm")
    res.lob_spread = deep_feats.get("lob_spread")
    res.lob_vwap_dev = deep_feats.get("lob_vwap_dev")
    res.lob_mom_change = deep_feats.get("lob_mom_change")

    # ── Z-score + Bayesian posteriors ─────────────────────────────────────────
    if strike and res.expected_move and res.expected_move > 0:
        res.distance = btc_price - strike
        res.z_score  = res.distance / res.expected_move
        # Fix #3: Student-T with df=6 for fat-tail correction (BTC kurtosis > 5).
        # normal_cdf systematically overestimates tail probabilities.
        res.posterior_fair_up   = student_t_cdf(res.z_score, df=6)
        res.posterior_fair_down = 1.0 - res.posterior_fair_up

    # ── Time-to-expiry probability decay ──────────────────────────────────────
    # As window approaches expiry, posterior should converge toward observed outcome.
    # With < 2 min remaining, the z-score becomes more deterministic — amplify conviction.
    # With > 10 min remaining, dampen conviction toward 0.5 (uncertainty is high).
    if res.posterior_fair_up is not None and minutes_remaining > 0:
        t_frac = clamp(t_frac, 0.0, 1.0)  # 1.0 at start, 0.0 at expiry
        if t_frac > 0.67:  # first third: dampen toward 0.5
            dampen = 0.7 + 0.3 * (1.0 - t_frac) / 0.33  # 0.7 at start → 1.0 at 67%
            res.posterior_fair_up = 0.5 + (res.posterior_fair_up - 0.5) * dampen
            res.posterior_fair_down = 1.0 - res.posterior_fair_up
        elif t_frac < 0.20:  # last 20%: amplify conviction
            amplify = 1.0 + 0.3 * (0.20 - t_frac) / 0.20  # 1.0 at 20% → 1.3 at expiry
            raw = 0.5 + (res.posterior_fair_up - 0.5) * amplify
            res.posterior_fair_up = clamp(raw, 1e-6, 1 - 1e-6)
            res.posterior_fair_down = 1.0 - res.posterior_fair_up

    # Bayesian update: blend fair value with market prior in logit space.
    # Fix #4: Signal weight reduced from 0.50 to 0.30 — market price is the stronger signal
    # (Manski 2004, Nofer 2018: prediction market prices aggregate thousands of traders' info).
    _bayes_signal_weight = float(getattr(Config, 'BAYES_SIGNAL_WEIGHT', 0.30))
    if res.posterior_fair_up is not None and yes_mid is not None:
        mkt_prior = clamp(yes_mid, 0.01, 0.99)
        signal_lo = logit(res.posterior_fair_up)
        po        = logit(mkt_prior) + _bayes_signal_weight * signal_lo
        up        = clamp(inv_logit(po), 1e-6, 1 - 1e-6)
        res.posterior_final_up   = up
        res.posterior_final_down = 1.0 - up
    elif res.posterior_fair_up is not None:
        res.posterior_final_up   = res.posterior_fair_up
        res.posterior_final_down = res.posterior_fair_down

    # ── Belief volatility ─────────────────────────────────────────────────────
    current_p = clamp(res.posterior_final_up or 0.5, 1e-6, 1 - 1e-6)
    current_x = logit(current_p)
    res.current_x = current_x

    # Update rolling belief-vol window
    cutoff = now_ts - Config.BELIEF_VOL_LOOKBACK_SEC
    samples = [s for s in state.belief_vol_samples if s.ts >= cutoff]
    if state.prev_x is not None:
        delta_x = current_x - state.prev_x
        if math.isfinite(delta_x):
            samples.append(BeliefVolSample(delta_x=delta_x, ts=now_ts))

    # Compute sigma_B
    if len(samples) >= 2:
        deltas = [s.delta_x for s in samples]
        mean   = sum(deltas) / len(deltas)
        var    = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
        res.sigma_b = min(math.sqrt(var), 1.0)  # Cap to prevent extreme multipliers
    else:
        res.sigma_b = Config.BELIEF_VOL_DEFAULT

    # Belief-vol multiplier
    time_factor = 1.0
    if Config.BELIEF_VOL_TIME_DECAY_ENABLED:
        window_min = Config.WINDOW_SEC / 60.0
        if window_min > 0:
            # Scale effect of belief volatility by fraction of window remaining.
            time_factor = clamp(minutes_remaining / window_min, 0.0, 1.0)

    raw_mult = 1.0 + (res.sigma_b - Config.BELIEF_VOL_DEFAULT) * time_factor
    max_mult = Config.BELIEF_VOL_LATE_MAX if minutes_remaining < 5.0 else Config.BELIEF_VOL_MULT_MAX
    res.bvol_multiplier = clamp(raw_mult, Config.BELIEF_VOL_MULT_MIN, max_mult)

    # Store updated samples back (caller must persist this to state)
    state.belief_vol_samples = samples
    state.prev_x             = current_x

    # ── Technical scoring ─────────────────────────────────────────────────────
    # Compute individual scores for logging / feature dict.
    # Final accumulation into `signed` happens via group-max below.

    # EMA crossover — continuous, ATR-normalized [-2, 2]
    if indic.ema9 is not None and indic.ema20 is not None and atr14 and atr14 > 0:
        ema_gap = indic.ema9 - indic.ema20
        res.ema_score = clamp(ema_gap / (atr14 * 0.002), -2.0, 2.0)

    # VWAP deviation — continuous, ATR-normalized [-1, 1]
    if indic.vwma15 is not None and atr14 and atr14 > 0:
        vwap_dev = btc_price - indic.vwma15
        res.vwap_score = clamp(vwap_dev / (atr14 * 0.5), -1.0, 1.0)

    # RSI — continuous, centered at 50, ratio-based [-2, 2]
    if indic.rsi14 is not None:
        res.rsi_score = clamp((50.0 - indic.rsi14) / 25.0, -2.0, 2.0)

    # ATR amplifier — scales with ATR magnitude, not binary
    if atr14 and atr14 > 0:
        atr_factor = clamp((atr14 - 120.0) / 80.0, 0.0, 1.0)  # ramps 120→200
        if res.ema_score != 0:
            res.atr_score = atr_factor * (1.0 if res.ema_score > 0 else -1.0)

    # MACD histogram — continuous, ATR-normalized [-1, 1]
    if indic.macd_hist is not None and atr14 and atr14 > 0:
        res.macd_score = clamp(indic.macd_hist / (atr14 * 0.01), -1.0, 1.0)

    # Stochastic — continuous, centered at 50 [-1.5, 1.5]
    if indic.stoch_k is not None:
        res.stoch_score = clamp((50.0 - indic.stoch_k) / 30.0, -1.0, 1.0)
        # Crossover bonus when oversold/overbought agrees with trend
        if indic.stoch_k < 20 and res.ema_score < 0:
            res.stoch_score = clamp(res.stoch_score - 0.5, -1.5, 1.5)
        elif indic.stoch_k > 80 and res.ema_score > 0:
            res.stoch_score = clamp(res.stoch_score + 0.5, -1.5, 1.5)

    # MFI divergence — strength-scaled by magnitude of MFI change [-2, 2]
    if indic.mfi14 is not None and state.prev_mfi is not None and state.prev_price is not None:
        mfi_change = abs(indic.mfi14 - state.prev_mfi)
        div_strength = clamp(mfi_change / 20.0, 0.1, 1.0)
        if btc_price > state.prev_price and indic.mfi14 < state.prev_mfi:
            res.mfi_score = -2.0 * div_strength   # bearish divergence
        elif btc_price < state.prev_price and indic.mfi14 > state.prev_mfi:
            res.mfi_score = 2.0 * div_strength    # bullish divergence

    # Multi-period OBV divergence — slope-magnitude scaled [-2.5, 2.5]
    if indic.obv_slope is not None and indic.price_slope is not None:
        p_sl = indic.price_slope
        o_sl = indic.obv_slope
        if p_sl > 0 and o_sl < 0:          # bearish divergence
            strength = clamp(abs(o_sl) / (abs(p_sl) + 1e-9), 0.2, 1.0)
            res.obv_score = -2.5 * strength
        elif p_sl < 0 and o_sl > 0:        # bullish divergence
            strength = clamp(abs(o_sl) / (abs(p_sl) + 1e-9), 0.2, 1.0)
            res.obv_score = 2.5 * strength
        elif (p_sl > 0 and o_sl > 0) or (p_sl < 0 and o_sl < 0):
            res.obv_score = 0.5 * (1.0 if p_sl > 0 else -1.0)  # confirmation

    # ── Microstructure ────────────────────────────────────────────────────────
    res.vpin_proxy      = vpin_proxy
    res.bid_depth       = bid_depth20
    res.ask_depth       = ask_depth20
    res.deep_imbalance  = deep_imbalance
    res.deep_ofi        = deep_ofi
    res.cvd             = true_cvd
    res.microprice      = microprice

    if is_stale_micro:
        # Use last-known scores
        cvd_score       = state.last_cvd_score
        ofi_score       = state.last_ofi_score
        imbalance_score = state.last_imbalance_score
        flow_accel      = state.last_flow_accel_score
    else:
        # Volume-weighted CVD scoring — continuous [-2, 2]
        if cvd_total_vol > 0 and cvd_delta != 0:
            vol_ratio = cvd_total_vol / max(prev_cvd_total_vol, 1e-6) if prev_cvd_total_vol > 0 else 1.0
            vol_multiplier = clamp(vol_ratio, 0.5, 2.0)
            cvd_score = clamp(cvd_delta * vol_multiplier, -2.0, 2.0)
        else:
            cvd_score = 0.0

        # OFI — continuous, depth-normalized [-2, 2]
        depth_vol = bid_depth20 + ask_depth20
        norm_ofi = deep_ofi / depth_vol if depth_vol > 0 else 0.0
        ofi_score = clamp(norm_ofi * 5.0, -2.0, 2.0)

        # Depth imbalance — continuous, centered at 0.5 [-1, 1]
        imbalance_score = clamp((deep_imbalance - 0.5) * 10.0, -1.0, 1.0)

        # Flow acceleration
        prev_ofi = state.prev_ofi_recent or 0.0
        if deep_ofi > prev_ofi:
            flow_accel = 1.0
        elif deep_ofi < prev_ofi:
            flow_accel = -1.0
        else:
            flow_accel = 0.0

    res.cvd_score        = cvd_score
    res.ofi_score        = ofi_score
    res.imbalance_score  = imbalance_score
    res.flow_accel_score = flow_accel

    # ── TOB imbalance — continuous, centered at 0.5 [-1.5, 1.5] ──────────────
    tob_score = clamp((tob_imbalance - 0.5) * 5.0, -1.5, 1.5)
    res.tob_score = tob_score

    # ── CVD velocity — continuous [-2, 2] ──────────────────────────────────────
    cvd_vel_score = clamp(cvd_velocity * 4.0, -2.0, 2.0)
    res.cvd_velocity_score = cvd_vel_score

    # ── Polymarket trade flow — continuous [-1.5, 1.5] ─────────────────────────
    pm_flow_score = clamp(pm_net_flow / 33.0, -1.5, 1.5)
    res.pm_flow_score = pm_flow_score

    # (Micro scores are accumulated via group-max in the final signed= block below)

    # ── Whale flow — continuous [-3, 3] with late-window boost ─────────────────
    whale_flow_score = clamp(whale_flow / 167.0, -3.0, 3.0)
    # Boost whale signal when within last 5 minutes (informed money more predictive near expiry)
    if minutes_remaining < 5.0 and whale_flow_score != 0.0:
        whale_flow_score = clamp(whale_flow_score * 1.5, -3.0, 3.0)
    res.whale_flow_score = whale_flow_score

    # ── Spread skew — continuous log-ratio [-2, 2] ─────────────────────────────
    spread_skew_score = 0.0
    yes_spread = (yes_ask - yes_bid) if yes_ask and yes_bid else None
    no_spread  = (no_ask  - no_bid)  if no_ask  and no_bid  else None
    if yes_spread and no_spread and yes_spread > 0 and no_spread > 0:
        skew_ratio = no_spread / yes_spread
        # log ratio: wide NO spread (high ratio) = market leans UP = positive score
        spread_skew_score = clamp(math.log(max(skew_ratio, 0.01)) * 1.5, -2.0, 2.0)
    res.spread_skew_score = spread_skew_score

    # ── Tier 1: Multi-Window Momentum ─────────────────────────────────────────
    window_momentum_score = 0.0
    outcomes = window_outcomes or []
    if len(outcomes) >= 2:
        recent = outcomes[-3:]  # last 3 windows
        ups = recent.count("UP")
        downs = recent.count("DOWN")
        if len(recent) >= 3 and ups == 3:
            window_momentum_score = 1.0
        elif len(recent) >= 3 and downs == 3:
            window_momentum_score = -1.0
        elif len(recent) >= 2 and ups >= 2:
            window_momentum_score = 0.5
        elif len(recent) >= 2 and downs >= 2:
            window_momentum_score = -0.5
    res.window_momentum_score = window_momentum_score

    # ── Tier 2: Liquidation Cascade ───────────────────────────────────────────
    liq_cascade_score = 0.0
    if liq_cascade > 2_000_000:      # >$2M long liquidations → bearish cascade
        liq_cascade_score = -3.0
    elif liq_cascade > 500_000:
        liq_cascade_score = -1.5
    elif liq_cascade < -2_000_000:   # >$2M short liquidations → bullish cascade
        liq_cascade_score = 3.0
    elif liq_cascade < -500_000:
        liq_cascade_score = 1.5
    res.liq_cascade_score = liq_cascade_score

    # ── Funding rate delta — continuous [-2, 2] ────────────────────────────────
    funding_delta_score = 0.0
    if funding_rate is not None and funding_rate_prev is not None:
        delta = funding_rate - funding_rate_prev
        # Negative: spike in funding = overleveraged longs = bearish
        funding_delta_score = clamp(-delta / 0.0002, -2.0, 2.0)
    res.funding_delta_score = funding_delta_score

    # ── Phase 2: NEW SIGNAL COMPONENTS ────────────────────────────────────────

    # Oracle Lag — removed (unreliable, Polymarket oracle divergence too noisy)
    res.oracle_lag_score = 0.0
    if oracle_px and oracle_px > 0:
        res.oracle_px = oracle_px
        if oracle_update_ts:
            res.oracle_update_age = now_ts - oracle_update_ts

    # Spread Pressure
    res.spread_pressure_score = 0.0
    spread_compression = 0.0
    if yes_ask is not None and yes_mid is not None and yes_mid > 0:
        yes_bid = (yes_mid * 2) - yes_ask
        if yes_bid > 0 and yes_ask > yes_bid:
            spread_pct = (yes_ask - yes_bid) / yes_bid
            if spread_pct < 0.015:
                spread_compression = 1.0
            elif spread_pct > 0.05:
                spread_compression = -0.5

    if spread_compression != 0.0:
        push_dir = 1.0 if imbalance_score > 0 else (-1.0 if imbalance_score < 0 else 0)
        res.spread_pressure_score = spread_compression * push_dir

    # Funding rate — removed as standalone (redundant with funding_delta in flow group)
    res.funding_rate_score = 0.0
    if funding_rate is not None:
        res.funding_rate = funding_rate

    # Liquidity vacuum — continuous depth ratio [-2, 2]
    liq_vacuum_score = 0.0
    if bid_depth20 > 0 and ask_depth20 > 0:
        total_depth = bid_depth20 + ask_depth20
        liq_vacuum_score = clamp((bid_depth20 - ask_depth20) / total_depth * 4.0, -2.0, 2.0)
    res.liq_vacuum_score = liq_vacuum_score

    # Bollinger Band position — continuous, centered at 0.5 [-1, 1]
    bb_position_score = 0.0
    if (indic.bb_upper is not None and indic.bb_lower is not None
            and indic.bb_upper != indic.bb_lower):
        bb_pos = (btc_price - indic.bb_lower) / (indic.bb_upper - indic.bb_lower)
        bb_pos = clamp(bb_pos, 0.0, 1.0)
        bb_position_score = clamp((bb_pos - 0.5) * 3.0, -1.0, 1.0)
    res.bb_position_score = bb_position_score

    # Cross-Exchange CVD — removed (±0.5 too small to matter, adds noise)
    res.cross_exch_score = 0.0

    # Accumulated OFI — continuous [-2, 2]
    accum_ofi_score = clamp(accumulated_ofi / Config.OFI_15M, -2.0, 2.0) if Config.OFI_15M > 0 else 0.0
    res.accum_ofi_score = accum_ofi_score

    # Prediction Market Mispricing Detection
    # NOTE: Mispricing is applied after direction is known so it can be aligned
    # with the intended trade side (YES/NO). Here we just initialise.
    res.misprice_score = 0.0

    # Multi-Timeframe Momentum
    mtf_momentum_score = 0.0
    if indic.vwma15 is not None and indic.ema9 is not None:
        price_above_vwma = btc_price > indic.vwma15
        price_above_ema9 = btc_price > indic.ema9
        if price_above_vwma and price_above_ema9:
            mtf_momentum_score = 1.0
        elif not price_above_vwma and not price_above_ema9:
            mtf_momentum_score = -1.0
    res.mtf_momentum_score = mtf_momentum_score

    # ADX + Stochastic Momentum Boost
    adx_stoch_boost = 0.0
    if indic.adx14 is not None and indic.stoch_k is not None and indic.stoch_d is not None:
        if indic.adx14 > 25 and indic.stoch_k > indic.stoch_d and indic.stoch_k < 80:
            adx_stoch_boost = 1.0
        elif indic.adx14 > 25 and indic.stoch_k < indic.stoch_d and indic.stoch_k > 20:
            adx_stoch_boost = -1.0
    res.adx_stoch_boost = adx_stoch_boost

    # OBI (computed here for use in group block and later penalty)
    obi_denom = total_bid_size + total_ask_size
    obi = (total_bid_size - total_ask_size) / obi_denom if obi_denom > 0 else 0.0
    res.obi = obi

    # ── Phase 4: Group-based scoring (prevents correlated inflation) ──────────
    # Each group contributes its maximum-magnitude element, not the sum.
    # This caps theoretical max at ~6 groups × ~3 each ≈ ±18 instead of ±30+.
    def _signed_max(*vals):
        """Return the value with the largest absolute magnitude."""
        filtered = [v for v in vals if v is not None]
        return max(filtered, key=abs) if filtered else 0.0

    # ── 4-group layout (absorbs former standalones + new_sig into proper groups)
    trend_group    = _signed_max(res.ema_score, res.vwap_score, res.macd_score,
                                 res.bb_position_score, res.mtf_momentum_score)
    momentum_group = _signed_max(res.rsi_score, res.stoch_score, res.mfi_score,
                                 res.obv_score, res.adx_stoch_boost)
    flow_group     = _signed_max(res.cvd_score, res.ofi_score, res.accum_ofi_score,
                                 res.whale_flow_score, res.funding_delta_score)
    micro_group    = _signed_max(res.imbalance_score, res.spread_pressure_score,
                                 res.liq_vacuum_score, res.spread_skew_score,
                                 res.liq_cascade_score)
    # Standalone timing modifiers (additive by design — micro timing, not directional)
    signed = (
        trend_group + momentum_group + flow_group + micro_group
        + tob_score * 0.6 + cvd_vel_score * 0.8 + pm_flow_score * 0.5
    )
    log.debug(
        f"Score groups: trend={trend_group:.2f} mom={momentum_group:.2f} "
        f"flow={flow_group:.2f} micro={micro_group:.2f} → {signed:.2f}"
    )

    # ── Post-group modifiers ──────────────────────────────────────────────────
    # Stale micro penalty
    if is_stale_micro and abs(signed) < 5.0:
        signed *= 0.8

    # Window-specific calibration
    if t_frac > 0.87:  # first ~2 min: dampen
        signed *= 0.75
    elif t_frac < 0.20:  # last ~3 min: add micro boost
        micro_boost = (tob_score * 0.6 + cvd_vel_score * 0.8) * 0.3
        signed += micro_boost

    # Belief-vol multiplier
    signed *= res.bvol_multiplier

    # Cap raw score to prevent extreme values from dominating the EMA
    signed = max(-8.0, min(8.0, signed))

    res.signed_score = signed
    # Phase 7: EMA smoothing to reduce single-cycle noise.
    # In the last 3 minutes, increase alpha so the score responds fast to market repricing.
    # Technical indicators lag; the late-window market price IS the signal.
    if state.prev_cycle_score is not None:
        # Clamp prev_score defensively — corrupted state or old builds can have out-of-range values
        prev_clamped = max(-8.0, min(8.0, state.prev_cycle_score))
        alpha = 0.85 if minutes_remaining < 3.0 else 0.6
        res.signed_score = alpha * res.signed_score + (1 - alpha) * prev_clamped
    # Final clamp: EMA output must also stay within ±8
    res.signed_score = max(-8.0, min(8.0, res.signed_score))
    res.abs_score    = abs(res.signed_score)

    # ── Phase 5: ML Inference ────────────────────────────────────────────────
    if inference_engine:
        feats = res.to_feature_dict()
        res.model_prob_up = inference_engine.predict_up_prob(feats)
        res.model_score = res.model_prob_up

    # Ensemble: blend Bayesian posterior (prior) with model probability every cycle.
    if res.posterior_final_up is not None and res.model_prob_up is not None:
        w_b = float(getattr(Config, "ENSEMBLE_BAYES_WEIGHT", 0.40) or 0.40)
        w_m = float(getattr(Config, "ENSEMBLE_MODEL_WEIGHT", 0.60) or 0.60)
        w_sum = w_b + w_m
        if w_sum <= 0:
            w_b, w_m, w_sum = 0.40, 0.60, 1.0
        p = (w_b * res.posterior_final_up + w_m * res.model_prob_up) / w_sum
        p = clamp(p, 1e-6, 1 - 1e-6)
        res.posterior_final_up = p
        res.posterior_final_down = 1.0 - p

    # Delta tracking for reporting
    if state.prev_cycle_score is not None:
        res.score_delta = res.signed_score - state.prev_cycle_score
    if state.prev_cycle_price is not None:
        res.price_delta = btc_price - state.prev_cycle_price

    # Direction from POSTERIOR
    if res.posterior_final_up is not None and res.posterior_final_down is not None:
        if res.posterior_final_up > res.posterior_final_down:
            res.direction = "UP"
        else:
            res.direction = "DOWN"
    else:
        if signed > 0:
            res.direction = "UP"
        elif signed < 0:
            res.direction = "DOWN"
        else:
            res.direction = "NEUTRAL"

    # Apply OBI penalty after direction known
    if res.direction == "UP" and obi < -0.3:
        res.signed_score -= 1.0
    elif res.direction == "DOWN" and obi > 0.3:
        res.signed_score -= 1.0
    res.abs_score = abs(res.signed_score)

    # Direction-aware Prediction Market Mispricing Detection (YES & NO)
    if (
        res.direction in ("UP", "DOWN")
        and res.posterior_final_up is not None
        and res.posterior_final_down is not None
        and yes_mid is not None
        and no_mid is not None
    ):
        if res.direction == "UP":
            model_p = res.posterior_final_up
            mkt_p = clamp(yes_mid, 0.01, 0.99)
        else:
            # For NO, compare DOWN probability to NO price
            model_p = res.posterior_final_down
            mkt_p = clamp(no_mid, 0.01, 0.99)

        edge_prob = model_p - mkt_p
        misprice_score = 0.0
        if abs(edge_prob) > 0.08:
            misprice_score = 1.5 * (1.0 if edge_prob > 0 else -1.0)
        elif abs(edge_prob) > 0.02:  # was 0.05 — calibrated for old flat 2% fee; real fee is ~0.09% at extremes
            misprice_score = 0.5 * (1.0 if edge_prob > 0 else -1.0)

        res.misprice_score = misprice_score
        if misprice_score != 0.0:
            res.signed_score += misprice_score
            res.abs_score = abs(res.signed_score)

    # Optional: Couple strong signed_score back into a lightly adjusted posterior.
    if (
        Config.SCORE_POSTERIOR_COUPLING_ENABLED
        and res.posterior_final_up is not None
        and res.posterior_final_down is not None
    ):
        # Normalise score into [-1, 1] bucket via division by 10
        score_term = clamp(res.signed_score / 10.0, -1.0, 1.0)
        raw_shift = Config.SCORE_POSTERIOR_COUPLING_K * score_term
        shift = clamp(raw_shift, -Config.SCORE_POSTERIOR_MAX_SHIFT, Config.SCORE_POSTERIOR_MAX_SHIFT)

        adj_up = clamp(res.posterior_final_up + shift, 1e-6, 1 - 1e-6)
        res.posterior_adj_up = adj_up
        res.posterior_adj_down = 1.0 - adj_up

    # ── Edge computation (fee-adjusted) ───────────────────────────────────────
    # Polymarket fee curve: fee_per_share = 0.0625 × p × (1-p)
    # Max fee ~1.56% at p=0.50; near-zero at extremes (e.g. p=0.985 → ~0.09%).
    # Crypto markets get a 20% maker rebate → effective taker fee = 0.0625 × p × (1-p).
    # This replaces the old flat 2% which massively overstated fees on extreme-priced contracts.
    def _pm_fee(p: float) -> float:
        """Polymarket taker fee per share at price p."""
        return 0.0625 * p * (1.0 - p)

    if res.posterior_final_up is not None and yes_ask is not None:
        fee_up = _pm_fee(yes_ask)
        res.edge_up = (res.posterior_final_up * (1.0 - fee_up)) - yes_ask
    if res.posterior_final_down is not None and no_ask is not None:
        fee_down = _pm_fee(no_ask)
        res.edge_down = (res.posterior_final_down * (1.0 - fee_down)) - no_ask

    if res.direction == "UP":
        res.target_side = "YES"
        res.target_edge = res.edge_up
    elif res.direction == "DOWN":
        res.target_side = "NO"
        res.target_edge = res.edge_down
    else:
        res.target_side = None
        res.target_edge = None

    # Monster detection
    chosen_posterior = (
        res.posterior_final_up   if res.direction == "UP" else
        res.posterior_final_down if res.direction == "DOWN" else 0.0
    ) or 0.0
    # Standard monster: high technical score + high posterior
    _standard_monster = (
        res.abs_score >= Config.MONSTER_SCORE and
        chosen_posterior >= Config.MONSTER_POSTERIOR
    )
    # Near-certain override: posterior >= 0.995 with any meaningful directional signal.
    # At 99.5%+ conviction the posterior is more reliable than a technical score of 8.0.
    _near_certain_override = (chosen_posterior >= 0.995 and res.abs_score >= 1.0)
    res.monster_signal = _standard_monster or _near_certain_override

    # ── Skip gates ────────────────────────────────────────────────────────────
    gates = []

    # Global distance gate (ATR-scaled)
    _min_dist_atr_mult = float(getattr(Config, "MIN_ENTRY_DISTANCE_ATR_MULT", 0.25))
    if indic.atr14 is not None and indic.atr14 > 0 and res.distance is not None:
        min_required_dist = _min_dist_atr_mult * indic.atr14
        if abs(res.distance) < min_required_dist and not res.monster_signal:
            gates.append(f"distance_too_close={abs(res.distance):.1f}_req={min_required_dist:.1f}")

    # ADX filter
    if indic.adx14 is not None:
        if indic.adx14 < Config.ADX_TREND_THRESHOLD and minutes_remaining > 5.0 and not res.monster_signal:
            gates.append(f"low_trend_adx={indic.adx14:.1f}")

    # Neutral direction
    if res.direction == "NEUTRAL":
        gates.append("neutral_direction")

    # Score-direction disagreement gate: block if technicals strongly contradict posterior.
    # direction is set by posterior; signed_score is from technicals.
    # If direction=DOWN but score is strongly positive (UP), the signals disagree.
    _score_disagrees = (
        (res.direction == "DOWN" and res.signed_score > 2.0) or
        (res.direction == "UP" and res.signed_score < -2.0)
    )

    if _score_disagrees and not res.monster_signal and t_frac <= 0.30:
        gates.append(f"score_direction_disagree={res.signed_score:+.1f}_vs_{res.direction}")

    # VPIN toxicity gate — exempts high-conviction signals (posterior ≥ 0.85)
    # since one-sided order books often accompany genuine directional moves.
    if (vpin_proxy > Config.VPIN_BLOCK_THRESHOLD and res.abs_score < 6.0
            and not res.monster_signal and chosen_posterior < 0.85):
        gates.append(f"vpin_toxic={vpin_proxy:.3f}")

    # Near-50/50 market gate: when both sides are priced between 0.35-0.65,
    # the market is genuinely uncertain. Only enter with very high conviction.
    _mkt_uncertain = (
        yes_mid is not None and no_mid is not None
        and 0.35 <= yes_mid <= 0.65 and 0.35 <= no_mid <= 0.65
    )
    if _mkt_uncertain and chosen_posterior < 0.95 and not res.monster_signal:
        gates.append(f"market_uncertain_5050=YES{yes_mid:.2f}_NO{no_mid:.2f}")

    # One-sided market gate: only trade when the market has clearly picked a side.
    # YES >= 0.75 (trade YES) or NO >= 0.75 (trade NO). Prevents entering ambiguous markets.
    # Require multi-cycle confirmation to filter transient spikes
    # (Pennock & Sami 2007: single-tick prediction market spikes are noise)
    _one_sided = (
        yes_mid is not None and no_mid is not None
        and (
            (res.direction == "UP" and yes_mid >= 0.75) or
            (res.direction == "DOWN" and no_mid >= 0.75)
        )
    )
    _one_sided_confirmed = _one_sided and one_sided_cycles >= Config.ONE_SIDED_CONFIRM_CYCLES
    # Convergence snipe bypass: when the model leads the market price by a large gap,
    # entering early captures the convergence move before the market catches up.
    # Bypass not_one_sided when: posterior >= 0.82 AND model-market gap >= 12pp AND BTC clearly positioned.
    # Ref: Tetlock (2004) — PM prices overshoot then revert; Hanson (2003) — LMSR implies predictable convergence.
    _conv_posterior_min = float(getattr(Config, "CONVERGENCE_GATE_POSTERIOR_MIN", 0.82))
    _conv_gap_min       = float(getattr(Config, "CONVERGENCE_GATE_GAP_MIN", 0.12))
    _conv_dist_atr_mult = float(getattr(Config, "LATE_CONVICTION_DISTANCE_ATR_MULT", 0.25))
    if indic.atr14 and indic.atr14 > 0:
        _conv_dist_min = _conv_dist_atr_mult * indic.atr14
    else:
        _conv_dist_min = float(getattr(Config, "LATE_CONVICTION_DISTANCE", 40.0))
    _market_px          = yes_mid if res.direction == "UP" else (no_mid if res.direction == "DOWN" else None)
    _model_market_gap   = abs(chosen_posterior - (_market_px or chosen_posterior))
    _convergence_snipe  = (
        not res.monster_signal
        and chosen_posterior >= _conv_posterior_min
        and _model_market_gap >= _conv_gap_min
        and res.distance is not None
        and abs(res.distance) >= _conv_dist_min
    )
    if not _one_sided_confirmed and chosen_posterior < 0.95 and not res.monster_signal and not _convergence_snipe:
        side_px = yes_mid if res.direction == "UP" else no_mid
        gates.append(f"not_one_sided={res.direction}_px={side_px:.2f}_need>=0.75_cycles={one_sided_cycles}")
    elif _convergence_snipe and not _one_sided_confirmed:
        log.info(
            "convergence_snipe: bypassing not_one_sided — posterior=%.3f market_px=%.2f gap=%.2f dist=%.1f",
            chosen_posterior, _market_px or 0, _model_market_gap, abs(res.distance or 0),
        )

    # DYNAMIC EDGE + MONSTER OVERRIDE
    market_price = yes_mid if res.direction == "UP" else (no_mid if res.direction == "DOWN" else None)
    best_posterior = max(res.posterior_final_up or 0, res.posterior_final_down or 0)

    if market_price is None:
        effective_required_edge = 0.018
    else:
        if best_posterior >= 0.95:                                              # near-certain sure-thing
            effective_required_edge = 0.003 if minutes_remaining < 3 else 0.005
        elif minutes_remaining < 2:
            effective_required_edge = 0.005
        elif minutes_remaining < 3 and best_posterior >= 0.80:                  # late + strong conviction
            effective_required_edge = 0.006
        elif market_price < 0.10 or market_price > 0.90:
            effective_required_edge = 0.007  # was 0.012 — high market_price means small absolute edge is still +EV
        else:
            effective_required_edge = 0.018  # was 0.035

    # Hawkes timing proxy: use trade-arrival clustering to allow more aggressive
    # late-window sniping when the next event is likely imminent.
    if arrival_ts_ms and minutes_remaining < 5.0:
        try:
            now_ms = int(now_ts) * 1000
            ts_list = [int(x) for x in list(arrival_ts_ms)[-100:] if x]
            ts_list = [x for x in ts_list if x > 0 and x <= now_ms]
            if len(ts_list) >= 5:
                # Base intensity from recent mean interarrival.
                dts = [(ts_list[i] - ts_list[i - 1]) / 1000.0 for i in range(1, len(ts_list))]
                avg_dt = sum(dts) / len(dts) if dts else 0.0
                mu = (1.0 / avg_dt) if avg_dt > 1e-6 else 0.0
                alpha = 0.5 * mu
                beta = 1.0
                lam = mu
                for t_ms in ts_list[-50:]:
                    dt = max((now_ms - t_ms) / 1000.0, 0.0)
                    lam += alpha * math.exp(-beta * dt)
                if lam > 1e-9:
                    res.hawkes_next_event_sec = 1.0 / lam
        except Exception:
            res.hawkes_next_event_sec = None

    sign_agrees = (
        (res.direction == "UP" and res.signed_score >= 0) or
        (res.direction == "DOWN" and res.signed_score <= 0)
    )
    if (
        minutes_remaining < 5.0
        and res.hawkes_next_event_sec is not None
        and res.hawkes_next_event_sec < 5.0
        and sign_agrees
    ):
        hawkes_edge = float(getattr(Config, "HAWKES_LATE_REQUIRED_EDGE", 0.003) or 0.003)
        effective_required_edge = min(effective_required_edge, hawkes_edge)
        res.required_edge = min(res.required_edge, hawkes_edge)

    # Ultra-late force
    if minutes_remaining < 1.0 and best_posterior >= 0.97:
        effective_required_edge = 0.0
        log.info("ULTRA_LATE_FORCE: last minute + 97%+ conviction → edge=0")

    # Cross-market perp basis filter: treat basis as an independent edge proxy.
    _basis_agrees = (
        perp_basis_pct is not None and (
            (res.direction == "UP" and perp_basis_pct > 0) or
            (res.direction == "DOWN" and perp_basis_pct < 0)
        )
    )
    _basis_override = bool(
        _basis_agrees
        and res.basis_edge is not None
        and res.basis_edge >= float(getattr(Config, "BASIS_EDGE_MIN", 0.008) or 0.008)
    )

    # Edge gate
    if res.target_edge is None or res.target_edge < effective_required_edge:
        if not _basis_override:
            gates.append(f"edge_insufficient={res.target_edge or 0:.4f}_req={effective_required_edge:.3f}")

    # Monster sure-thing override
    if res.monster_signal and best_posterior >= 0.95:
        gates = [g for g in gates if "edge_insufficient" not in g]
        log.info(f"MONSTER_SURE_THING_OVERRIDE: posterior={best_posterior:.4f} → forcing trade")

    # Late-window conviction override — wires the LATE_CONVICTION_* config constants.
    # When BTC is clearly on one side of strike with high posterior near expiry,
    # technical scores (EMA, MACD) are lagging 5m data and the market has repriced.
    # Instead of fully suppressing the score gate, we:
    #   1. Relax the score threshold from 2.5 → LATE_CONVICTION_MIN_SCORE
    #   2. Bolster the score with late-window micro signals (OBI, CVD, OBV divergence)
    #   3. Relax the edge gate
    if indic.atr14 and indic.atr14 > 0:
        _late_dist_req = float(getattr(Config, "LATE_CONVICTION_DISTANCE_ATR_MULT", 0.25)) * indic.atr14
    else:
        _late_dist_req = Config.LATE_CONVICTION_DISTANCE

    is_late_conviction = (
        minutes_remaining <= Config.LATE_CONVICTION_MIN_REM
        and best_posterior >= Config.LATE_CONVICTION_POSTERIOR
        and res.distance is not None
        and abs(res.distance) >= _late_dist_req
    )

    # Late-conviction micro bolster: sum of confirming flow/micro signals.
    # These are real-time and still accurate in the last 3 minutes, unlike EMA/MACD.
    late_micro_boost = 0.0
    if is_late_conviction:
        # OBI: real-time order book imbalance (updated every cycle)
        if res.obi is not None and abs(res.obi) > 0.15:
            obi_dir = 1.0 if res.obi > 0 else -1.0
            score_dir = 1.0 if res.direction == "UP" else -1.0
            if obi_dir == score_dir:
                late_micro_boost += 0.5
        # CVD: real-time cumulative volume delta
        if abs(res.cvd_score) >= 0.5:
            cvd_dir = 1.0 if res.cvd_score > 0 else -1.0
            score_dir = 1.0 if res.direction == "UP" else -1.0
            if cvd_dir == score_dir:
                late_micro_boost += 0.5
        # OBV divergence: if OBV agrees with the trade direction
        if abs(res.obv_score) >= 0.5:
            obv_dir = 1.0 if res.obv_score > 0 else -1.0
            score_dir = 1.0 if res.direction == "UP" else -1.0
            if obv_dir == score_dir:
                late_micro_boost += 0.5

        # Relax the edge gate to the late-conviction threshold
        gates = [g for g in gates if "edge_insufficient" not in g]
        if res.target_edge is not None and res.target_edge < Config.LATE_CONVICTION_EDGE:
            gates.append(
                f"edge_insufficient={res.target_edge:.4f}_req={Config.LATE_CONVICTION_EDGE:.3f}"
            )
        log.info(
            f"LATE_CONVICTION: rem={minutes_remaining:.1f}m post={best_posterior:.3f} "
            f"dist={res.distance:.1f} micro_boost={late_micro_boost:.1f} "
            f"→ score_req relaxed to 0.5, edge relaxed to {Config.LATE_CONVICTION_EDGE:.3f}"
        )

    # Score gate — relaxed for late-conviction
    if is_late_conviction:
        late_effective_score = res.abs_score + late_micro_boost
        late_min_score = float(getattr(Config, "LATE_CONVICTION_MIN_SCORE", 1.25))
        if late_effective_score < late_min_score:
            gates.append(f"score_low={late_effective_score:.2f}_req={late_min_score:.1f}_late")
    elif res.abs_score < res.min_score and not res.monster_signal:
        gates.append(f"score_low={res.abs_score:.2f}_req={res.min_score:.1f}")

    # Score stability gate: require score to be above threshold for 2+ consecutive cycles.
    # A score that was below threshold last cycle (oscillating signal) has IC ≈ 0 — don't trade.
    # Exempts monster signals and cases where prev_cycle_score is unavailable.
    _stability_cycles = getattr(Config, "SCORE_STABILITY_MIN_CYCLES", 2)
    if (
        _stability_cycles >= 2
        and not res.monster_signal
        and state.prev_cycle_score is not None
        and abs(state.prev_cycle_score) < res.min_score
        and res.abs_score >= res.min_score  # current is above, prev was below
    ):
        gates.append(f"score_unstable=prev_{state.prev_cycle_score:.2f}_curr_{res.signed_score:.2f}")

    # Early window guard
    early_rem_threshold = (Config.WINDOW_SEC / 60.0) - Config.EARLY_WINDOW_GUARD_MIN
    if minutes_remaining > early_rem_threshold and not res.monster_signal:
        if res.abs_score < 7.0 and chosen_posterior < 0.90:
            gates.append(f"early_window_rem={minutes_remaining:.1f}min")

    # Monster too early
    if res.monster_signal and minutes_remaining > 11.0:
        gates.append(f"monster_too_early_rem={minutes_remaining:.1f}min")

    # Bollinger squeeze (regime-adaptive threshold)
    if indic.bb_width is not None and not res.monster_signal:
        if res.regime == "low":
            bb_thresh = Config.BB_SQUEEZE_LOW
        elif res.regime == "high":
            bb_thresh = Config.BB_SQUEEZE_HIGH
        else:
            bb_thresh = Config.BB_SQUEEZE_NORMAL
        if indic.bb_width < bb_thresh:
            gates.append(f"bb_squeeze={indic.bb_width:.4f}_thresh={bb_thresh:.4f}")

    # VOLUME CONFIRMATION FOR EXTREMES
    if best_posterior > 0.95 and cvd_total_vol <= 8.0:
        gates.append("low_volume_on_sure_thing")
        log.info(f"LOW_VOLUME_BLOCK: volume={cvd_total_vol:.1f} on 95%+ conviction")

    # Preferred trading time gate
    if (
        not Config.is_preferred_trading_time()
        and chosen_posterior < float(getattr(Config, "OUTSIDE_HOURS_ENTRY_POSTERIOR_MIN", 0.75) or 0.75)
    ):
        gates.append('outside_preferred_hours')

    # BTC momentum velocity gate: block entry when BTC surges against trade direction in last 15s.
    # Uses 3-cycle (15s at 5s/cycle) velocity in ATR-normalized units.
    # Ref: Almgren & Chriss (2001) — adverse selection cost spikes when momentum opposes direction.
    _prev_px3 = getattr(state, "prev_price3", None)
    if _prev_px3 is not None and indic.atr14 is not None and indic.atr14 > 0 and not res.monster_signal:
        _velocity_15s = (btc_price - _prev_px3) / indic.atr14  # signed, in ATR units
        _mom_dir_sign = 1 if res.direction == "UP" else -1
        _adverse_vel = -_mom_dir_sign * _velocity_15s           # positive = BTC moving against trade
        _momentum_thresh = float(getattr(Config, "MOMENTUM_GATE_ATR_THRESHOLD", 0.25))
        if _adverse_vel > _momentum_thresh:
            gates.append(f"btc_momentum_adverse={_velocity_15s:+.3f}ATR/15s")

    # Polymarket LOB adverse imbalance gate: heavy ask side = informed sellers distributing.
    # Ref: Cont, Kukanov & Stoikov (2014) — LOB imbalance predicts short-term price direction.
    if total_bid_size + total_ask_size > 0 and not res.monster_signal:
        _pm_ask_ratio = total_ask_size / (total_bid_size + total_ask_size)
        _pm_lob_thresh = float(getattr(Config, "PM_LOB_ADVERSE_THRESHOLD", 0.80))
        if _pm_ask_ratio > _pm_lob_thresh:
            gates.append(f"pm_lob_adverse_ask_ratio={_pm_ask_ratio:.2f}")

    # Cross-asset funding rate gate: positive funding opposes a DOWN trade (bullish market bias).
    # Ref: Liu & Tsyvinski (2021) — funding rate has 24h directional predictive power on BTC.
    if funding_rate is not None and res.direction is not None and not res.monster_signal:
        _f_dir_sign = 1 if res.direction == "UP" else -1
        _funding_adverse = -_f_dir_sign * funding_rate  # positive = funding opposes our direction
        _funding_thresh = float(getattr(Config, "FUNDING_RATE_GATE_THRESHOLD", 0.0002))
        if _funding_adverse > _funding_thresh:
            gates.append(f"funding_rate_adverse={funding_rate:+.5f}")

    # PM price surge gate: block entry if the target side has surged > PM_PRICE_SURGE_GATE since last cycle.
    # A >12% move in one 5-second cycle means the convergence opportunity has already been consumed —
    # entering now buys the post-convergence peak, not the mispricing.
    # Ref: complementary to convergence_snipe — enter EARLY (at gap), block LATE (after surge consumed).
    _prev_mid = getattr(state, "prev_cycle_mid", None)
    _cur_side_mid = yes_mid if res.direction == "UP" else (no_mid if res.direction == "DOWN" else None)
    if (
        _prev_mid is not None and _cur_side_mid is not None and _prev_mid > 0
        and not res.monster_signal and not _convergence_snipe
    ):
        _pm_surge_pct = (_cur_side_mid - _prev_mid) / _prev_mid
        _pm_surge_thresh = float(getattr(Config, "PM_PRICE_SURGE_GATE", 0.12))
        if _pm_surge_pct > _pm_surge_thresh:
            gates.append(f"pm_price_surge={_pm_surge_pct*100:.1f}%_prev={_prev_mid:.2f}_cur={_cur_side_mid:.2f}")

    # Fix #6: Higher-timeframe trend conflict gate.
    # Block entries when the 1H EMA trend explicitly opposes our trade direction.
    # Only apply when we have a definitive UP/DOWN trend — NEUTRAL is permissive.
    # Ref: Jegadeesh & Titman (1993) — momentum persistence across timeframes.
    _htf_enabled = bool(getattr(Config, "HTF_TREND_GATE_ENABLED", True))
    if _htf_enabled and htf_trend != "NEUTRAL" and res.direction is not None and not res.monster_signal:
        _htf_opposes = (
            (res.direction == "UP" and htf_trend == "DOWN")
            or (res.direction == "DOWN" and htf_trend == "UP")
        )
        if _htf_opposes:
            gates.append(f"htf_trend_conflict:1H={htf_trend}_trade={res.direction}")

    # Fix #9: VPOC proximity gate.
    # Block entries when BTC price is too far from the volume-weighted fair value.
    # Large VPOC gaps indicate thin-volume price discovery — adversely selected fills.
    # Ref: Market Profile theory (Steidlmayer 1984) — VPOC acts as a market magnet.
    _vpoc_gate_pct = float(getattr(Config, "VPOC_PROXIMITY_GATE_PCT", 0.015))
    if vpoc is not None and vpoc > 0 and not res.monster_signal:
        _vpoc_dist = abs(btc_price - vpoc) / vpoc
        if _vpoc_dist > _vpoc_gate_pct:
            gates.append(f"vpoc_too_far={_vpoc_dist*100:.1f}%_from_{vpoc:.0f}")

    # Fix #10: Bearish candlestick pattern gate.
    # A fresh bearish engulfing or shooting star on the last 15m candle signals reversal risk
    # that overrides momentum on UP entries, and vice versa for DOWN entries.
    # Ref: Nison (1991) — high-reliability reversal signals in liquid markets.
    if candle_patterns is not None and res.direction is not None and not res.monster_signal:
        _bearish = candle_patterns.get("bearish_engulfing") or candle_patterns.get("shooting_star")
        _bullish = candle_patterns.get("bullish_engulfing") or candle_patterns.get("hammer")
        if res.direction == "UP" and _bearish:
            gates.append("candle_bearish_pattern:engulf_or_star_on_UP_entry")
        elif res.direction == "DOWN" and _bullish:
            gates.append("candle_bullish_pattern:engulf_or_hammer_on_DOWN_entry")

    res.skip_gates = gates

    # Structured gate evaluation log highlighting the primary blocking reason.
    primary_gate = gates[0] if gates else "CLEAR"
    edge_val = res.target_edge or 0.0
    log.info(
        "gate_eval: dir=%s primary=%s edge=%.4f req_edge=%.3f "
        "score=%.2f req_score=%.1f minutes_rem=%.1f",
        res.direction,
        primary_gate,
        edge_val,
        effective_required_edge,
        res.abs_score,
        res.min_score,
        minutes_remaining,
    )

    # ── Volatility surface (logging-only) ──────────────────────────────────
    res.vol_surface_adj = _compute_vol_surface(yes_mid, atr14, minutes_remaining)

    p_up_str = f"{res.posterior_final_up:.4f}" if res.posterior_final_up else "N/A"
    e_str = f"{res.target_edge:.4f}" if res.target_edge else "N/A"
    log.info(
        f"signal: score={res.signed_score:.2f} dir={res.direction} "
        f"postUp={p_up_str} edge={e_str} "
        f"regime={res.regime} monster={res.monster_signal} "
        f"gates={gates or 'CLEAR'}"
    )
    return res
