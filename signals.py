"""
Feature calculation, score weighting, and Bayesian momentum signals.
Extracted from logic.py during Phase 3 refactor.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from indicators import Indicators, normal_cdf, logit, inv_logit, clamp
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

    # Edge
    edge_up:              Optional[float] = None
    edge_down:            Optional[float] = None
    target_edge:          Optional[float] = None
    target_side:          Optional[str]   = None   # "YES" | "NO"

    # Thresholds (regime-adaptive)
    required_edge:        float  = 0.035
    min_score:            float  = 4.0

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
    # Phase 3: TOB + CVD velocity + Polymarket flow
    tob_imbalance:    float = 0.5,
    cvd_velocity:     float = 0.0,
    pm_net_flow:      float = 0.0,
    # Polymarket
    yes_mid:          Optional[float] = None,
    no_mid:           Optional[float] = None,
    yes_ask:          Optional[float] = None,
    no_ask:           Optional[float] = None,
    total_bid_size:   float = 0.0,
    total_ask_size:   float = 0.0,
    # Phase 5
    inference_engine: Optional[object] = None,
    # Phase 4 Optimization
    score_offset:     float = 0.0,
    edge_offset:      float = 0.0,
) -> SignalResult:

    res = SignalResult()

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

    res.required_edge, res.min_score = Config.get_regime_thresholds(atr14)
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

    # ── Z-score + Bayesian posteriors ─────────────────────────────────────────
    if strike and res.expected_move and res.expected_move > 0:
        res.distance = btc_price - strike
        res.z_score  = res.distance / res.expected_move
        res.posterior_fair_up   = normal_cdf(res.z_score)
        res.posterior_fair_down = 1.0 - res.posterior_fair_up

    # ── Time-to-expiry probability decay ──────────────────────────────────────
    # As window approaches expiry, posterior should converge toward observed outcome.
    # With < 2 min remaining, the z-score becomes more deterministic — amplify conviction.
    # With > 10 min remaining, dampen conviction toward 0.5 (uncertainty is high).
    if res.posterior_fair_up is not None and minutes_remaining > 0:
        window_min = Config.WINDOW_SEC / 60.0
        t_frac = minutes_remaining / window_min  # 1.0 at start, 0.0 at expiry
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
    # Signal weight 0.7 (was 0.5) — our model should dominate over market noise.
    if res.posterior_fair_up is not None and yes_mid is not None:
        mkt_prior = clamp(yes_mid, 0.01, 0.99)
        signal_lo = logit(res.posterior_fair_up)
        po        = logit(mkt_prior) + 0.7 * signal_lo
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
        res.sigma_b = math.sqrt(var)
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
    signed = 0.0

    # EMA crossover ±2
    if indic.ema9 is not None and indic.ema20 is not None:
        res.ema_score = 2.0 if indic.ema9 > indic.ema20 else -2.0
        signed += res.ema_score

    # VWAP ±1
    if indic.vwma15 is not None:
        res.vwap_score = 1.0 if btc_price > indic.vwma15 else -1.0
        signed += res.vwap_score

    # RSI
    if indic.rsi14 is not None:
        if indic.rsi14 > 70:
            res.rsi_score = 1.0
        elif indic.rsi14 < 30:
            res.rsi_score = -1.0
        signed += res.rsi_score

    # ATR amplifier ±1
    if atr14 and atr14 > 120.0:
        if res.ema_score > 0:
            res.atr_score = 1.0
        elif res.ema_score < 0:
            res.atr_score = -1.0
        signed += res.atr_score

    # MACD histogram
    if indic.macd_hist is not None:
        if indic.macd_hist > 0:
            res.macd_score = 1.0
        elif indic.macd_hist < 0:
            res.macd_score = -1.0
        signed += res.macd_score

    # Stochastic
    if indic.stoch_k is not None:
        if indic.stoch_k > 80:
            res.stoch_score = 1.0
        elif indic.stoch_k < 20:
            res.stoch_score = -1.0
        # Crossover bonus
        if indic.stoch_k < 20 and res.ema_score < 0:
            res.stoch_score -= 0.5
        elif indic.stoch_k > 80 and res.ema_score > 0:
            res.stoch_score += 0.5
        signed += res.stoch_score

    # MFI divergence
    if indic.mfi14 is not None and state.prev_mfi is not None and state.prev_price is not None:
        if btc_price > state.prev_price and indic.mfi14 < state.prev_mfi:
            res.mfi_score = -2.0   # bearish divergence
        elif btc_price < state.prev_price and indic.mfi14 > state.prev_mfi:
            res.mfi_score = 2.0    # bullish divergence
        signed += res.mfi_score

    # Multi-period OBV divergence
    if indic.obv_slope is not None and indic.price_slope is not None:
        p_sl = indic.price_slope
        o_sl = indic.obv_slope
        if p_sl > 0 and o_sl < 0:
            res.obv_score = -2.5    # bearish divergence
        elif p_sl < 0 and o_sl > 0:
            res.obv_score = 2.5     # bullish divergence
        elif (p_sl > 0 and o_sl > 0) or (p_sl < 0 and o_sl < 0):
            res.obv_score = 0.5 * (1.0 if p_sl > 0 else -1.0)  # confirmation
        signed += res.obv_score

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
        # Volume-weighted CVD scoring
        if cvd_total_vol > 0 and cvd_delta != 0:
            vol_ratio = cvd_total_vol / max(prev_cvd_total_vol, 1e-6) if prev_cvd_total_vol > 0 else 1.0
            vol_multiplier = clamp(vol_ratio, 0.5, 2.0)
            if cvd_delta > 0:
                cvd_score = 1.0 * vol_multiplier
            else:
                cvd_score = -1.0 * vol_multiplier
        else:
            cvd_score = 0.0

        # OFI (Phase 2 Normalized)
        depth_vol = bid_depth20 + ask_depth20
        norm_ofi = deep_ofi / depth_vol if depth_vol > 0 else 0.0
        
        if norm_ofi > 0.4:
            ofi_score = 2.0
        elif norm_ofi < -0.4:
            ofi_score = -2.0
        elif norm_ofi > 0.15:
            ofi_score = 1.0
        elif norm_ofi < -0.15:
            ofi_score = -1.0
        else:
            ofi_score = 0.0

        # Depth imbalance
        if deep_imbalance > 0.60:
            imbalance_score = 1.0
        elif deep_imbalance < 0.40:
            imbalance_score = -1.0
        else:
            imbalance_score = 0.0

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

    # ── Phase 3: TOB imbalance (level-1, separate from 20-level depth) ────────
    tob_score = 0.0
    if tob_imbalance > 0.65:
        tob_score = 1.5
    elif tob_imbalance < 0.35:
        tob_score = -1.5
    elif tob_imbalance > 0.55:
        tob_score = 0.5
    elif tob_imbalance < 0.45:
        tob_score = -0.5
    res.tob_score = tob_score

    # ── Phase 3: CVD velocity (linear regression slope, units: BTC/sec) ─────
    cvd_vel_score = 0.0
    if abs(cvd_velocity) > 0.5:        # strong velocity
        cvd_vel_score = 2.0 if cvd_velocity > 0 else -2.0
    elif abs(cvd_velocity) > 0.15:     # moderate velocity
        cvd_vel_score = 1.0 if cvd_velocity > 0 else -1.0
    res.cvd_velocity_score = cvd_vel_score

    # ── Phase 3: Polymarket trade flow signal ──────────────────────────────────
    pm_flow_score = 0.0
    if abs(pm_net_flow) > 50:        # strong directional flow
        pm_flow_score = 1.5 if pm_net_flow > 0 else -1.5
    elif abs(pm_net_flow) > 20:      # moderate flow
        pm_flow_score = 0.7 if pm_net_flow > 0 else -0.7
    res.pm_flow_score = pm_flow_score

    # Weighted micro (includes Phase 3 TOB + CVD velocity + PM flow)
    signed += cvd_score * 0.7 + ofi_score + imbalance_score + flow_accel
    signed += tob_score * 0.6 + cvd_vel_score * 0.8 + pm_flow_score * 0.5

    # ── Phase 2: NEW SIGNAL COMPONENTS ────────────────────────────────────────

    # Oracle Lag Detection
    oracle_lag_score = 0.0
    if oracle_px and oracle_px > 0 and btc_price > 0:
        res.oracle_px = oracle_px
        if oracle_update_ts:
            res.oracle_update_age = now_ts - oracle_update_ts
            
        divergence_pct = (btc_price - oracle_px) / oracle_px
        if divergence_pct > 0.0015:  # +0.15%
            oracle_lag_score = 2.0
        elif divergence_pct < -0.0015:
            oracle_lag_score = -2.0
    res.oracle_lag_score = oracle_lag_score
    signed += oracle_lag_score

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
    signed += res.spread_pressure_score

    # Perpetual Funding Rate Divergence
    res.funding_rate_score = 0.0
    if funding_rate is not None:
        res.funding_rate = funding_rate
        if funding_rate > 0.0003: # High positive -> bearish
            res.funding_rate_score = -1.5
        elif funding_rate > 0.00015:
            res.funding_rate_score = -0.5
        elif funding_rate < -0.0003: # High negative -> bullish
            res.funding_rate_score = 1.5
        elif funding_rate < -0.00015:
            res.funding_rate_score = 0.5
            
    signed += res.funding_rate_score

    # Liquidity Vacuum Detection
    liq_vacuum_score = 0.0
    if bid_depth20 > 0 and ask_depth20 > 0:
        ask_bid_ratio = ask_depth20 / bid_depth20
        bid_ask_ratio = bid_depth20 / ask_depth20
        if ask_bid_ratio < 0.30:        # thin asks → likely upward push
            liq_vacuum_score = 2.0
        elif ask_bid_ratio < 0.50:
            liq_vacuum_score = 1.0
        elif bid_ask_ratio < 0.30:       # thin bids → likely downward push
            liq_vacuum_score = -2.0
        elif bid_ask_ratio < 0.50:
            liq_vacuum_score = -1.0
    res.liq_vacuum_score = liq_vacuum_score
    signed += liq_vacuum_score * 0.8

    # Bollinger Band Position
    bb_position_score = 0.0
    if (indic.bb_upper is not None and indic.bb_lower is not None
            and indic.bb_upper != indic.bb_lower):
        bb_pos = (btc_price - indic.bb_lower) / (indic.bb_upper - indic.bb_lower)
        bb_pos = clamp(bb_pos, 0.0, 1.0)
        if bb_pos > 0.85:
            bb_position_score = 1.0
        elif bb_pos < 0.15:
            bb_position_score = -1.0
        elif bb_pos > 0.70:
            bb_position_score = 0.5
        elif bb_pos < 0.30:
            bb_position_score = -0.5
    res.bb_position_score = bb_position_score
    signed += bb_position_score

    # Cross-Exchange CVD Confirmation
    cross_exch_score = 0.0
    if cross_cvd_agree:
        cross_exch_score = 0.5 * (1.0 if cvd_delta > 0 else (-1.0 if cvd_delta < 0 else 0.0))
    else:
        cross_exch_score = -0.3 * (1.0 if signed > 0 else -1.0)
    res.cross_exch_score = cross_exch_score
    signed += cross_exch_score

    # Accumulated OFI Score
    accum_ofi_score = 0.0
    if abs(accumulated_ofi) > Config.OFI_15M:
        accum_ofi_score = 2.0 if accumulated_ofi > 0 else -2.0
    elif abs(accumulated_ofi) > Config.OFI_STRONG:
        accum_ofi_score = 1.0 if accumulated_ofi > 0 else -1.0
    res.accum_ofi_score = accum_ofi_score
    signed += accum_ofi_score * 0.8

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
    signed += mtf_momentum_score * 0.6

    # ADX + Stochastic Momentum Boost
    adx_stoch_boost = 0.0
    if indic.adx14 is not None and indic.stoch_k is not None and indic.stoch_d is not None:
        if indic.adx14 > 25 and indic.stoch_k > indic.stoch_d and indic.stoch_k < 80:
            adx_stoch_boost = 1.0
        elif indic.adx14 > 25 and indic.stoch_k < indic.stoch_d and indic.stoch_k > 20:
            adx_stoch_boost = -1.0
    res.adx_stoch_boost = adx_stoch_boost
    signed += adx_stoch_boost

    # Stale micro penalty
    if is_stale_micro and abs(signed) < 5.0:
        signed *= 0.8

    # ── Phase 3: Window-specific signal calibration ───────────────────────────
    # Early in the window (first 2 min): signals are noisy, dampen non-micro scores.
    # Late in the window (last 3 min): micro signals dominate, boost them.
    window_min = Config.WINDOW_SEC / 60.0
    t_frac = minutes_remaining / window_min if window_min > 0 else 0.5
    if t_frac > 0.87:  # first ~2 min of 15 min window
        # Dampen all technical scores but preserve micro
        signed *= 0.75
    elif t_frac < 0.20:  # last ~3 min
        # Late phase: micro signals are more predictive — boost their contribution
        micro_boost = (tob_score * 0.6 + cvd_vel_score * 0.8) * 0.3  # 30% extra
        signed += micro_boost

    # Apply belief-vol multiplier
    signed *= res.bvol_multiplier

    # OBI penalty
    obi_denom = total_bid_size + total_ask_size
    obi = (total_bid_size - total_ask_size) / obi_denom if obi_denom > 0 else 0.0
    res.obi = obi

    res.signed_score = signed
    res.abs_score    = abs(signed)

    # ── Phase 5: ML Inference ────────────────────────────────────────────────
    if inference_engine:
        feats = res.to_feature_dict()
        res.model_score = inference_engine.predict_up_prob(feats)
        if res.model_score is not None:
            log.info(f"ML Inference UP probability: {res.model_score:.4f}")
            if res.model_score > 0.8:
                res.signed_score += 1.0
            elif res.model_score < 0.2:
                res.signed_score -= 1.0

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
        if abs(edge_prob) > 0.10:
            misprice_score = 1.5 * (1.0 if edge_prob > 0 else -1.0)
        elif abs(edge_prob) > 0.05:
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
    fee_rate = 0.02
    if res.posterior_final_up is not None and yes_ask is not None:
        ev_win_up = 1.0 - fee_rate * (1.0 - yes_ask)
        res.edge_up = (res.posterior_final_up * ev_win_up) - yes_ask
    if res.posterior_final_down is not None and no_ask is not None:
        ev_win_down = 1.0 - fee_rate * (1.0 - no_ask)
        res.edge_down = (res.posterior_final_down * ev_win_down) - no_ask

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
    res.monster_signal = (
        res.abs_score >= Config.MONSTER_SCORE or
        chosen_posterior >= Config.MONSTER_POSTERIOR
    )

    # ── Skip gates ────────────────────────────────────────────────────────────
    gates = []

    # ADX filter
    if indic.adx14 is not None:
        if indic.adx14 < Config.ADX_TREND_THRESHOLD and minutes_remaining > 5.0 and not res.monster_signal:
            gates.append(f"low_trend_adx={indic.adx14:.1f}")

    # Neutral direction
    if res.direction == "NEUTRAL":
        gates.append("neutral_direction")

    # VPIN toxicity gate
    if vpin_proxy > Config.VPIN_BLOCK_THRESHOLD and res.abs_score < 4.0 and not res.monster_signal:
        gates.append(f"vpin_toxic={vpin_proxy:.3f}")

    # DYNAMIC EDGE + MONSTER OVERRIDE
    market_price = yes_mid if res.direction == "UP" else (no_mid if res.direction == "DOWN" else None)
    best_posterior = max(res.posterior_final_up or 0, res.posterior_final_down or 0)

    if market_price is None:
        effective_required_edge = 0.035
    else:
        if best_posterior >= 0.95:                                              # monster sure-thing
            effective_required_edge = 0.005 if minutes_remaining < 3 else 0.012
        elif minutes_remaining < 2:
            effective_required_edge = 0.010
        elif market_price < 0.10 or market_price > 0.90:
            effective_required_edge = 0.012
        else:
            effective_required_edge = 0.035

    # Ultra-late force
    if minutes_remaining < 1.0 and best_posterior >= 0.97:
        effective_required_edge = 0.0
        log.info("ULTRA_LATE_FORCE: last minute + 97%+ conviction → edge=0")

    # Edge gate
    if res.target_edge is None or res.target_edge < effective_required_edge:
        gates.append(f"edge_insufficient={res.target_edge or 0:.4f}_req={effective_required_edge:.3f}")

    # Monster sure-thing override
    if res.monster_signal and best_posterior >= 0.95:
        gates = [g for g in gates if "edge_insufficient" not in g]
        log.info(f"MONSTER_SURE_THING_OVERRIDE: posterior={best_posterior:.4f} → forcing trade")

    # Score gate
    if res.abs_score < res.min_score and not res.monster_signal:
        gates.append(f"score_low={res.abs_score:.2f}_req={res.min_score:.1f}")

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

    p_up_str = f"{res.posterior_final_up:.4f}" if res.posterior_final_up else "N/A"
    e_str = f"{res.target_edge:.4f}" if res.target_edge else "N/A"
    log.info(
        f"signal: score={res.signed_score:.2f} dir={res.direction} "
        f"postUp={p_up_str} edge={e_str} "
        f"regime={res.regime} monster={res.monster_signal} "
        f"gates={gates or 'CLEAR'}"
    )
    return res
