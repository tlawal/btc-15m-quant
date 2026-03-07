"""
Phase 5: Prometheus metrics exporter for time-series signal history and engine metrics.
"""
from prometheus_client import start_http_server, Gauge, Counter
import logging

log = logging.getLogger("metrics_exporter")

# System gauges
metric_wallet_balance = Gauge('btc15m_wallet_usdc', 'Wallet USDC Balance')
metric_unclaimed_winnings = Gauge('btc15m_unclaimed_usdc', 'Unclaimed PM Winnings')
metric_exposure = Gauge('btc15m_exposure_usd', 'Current exposure in USDC')

# Performance metrics
metric_win_rate = Gauge('btc15m_win_rate', 'Bot Win Rate (0-1)')
metric_profit_factor = Gauge('btc15m_profit_factor', 'Profit Factor')
metric_sharpe = Gauge('btc15m_sharpe_ratio', 'Sharpe Ratio')
metric_loss_streak = Gauge('btc15m_loss_streak', 'Current loss streak')
metric_total_pnl = Gauge('btc15m_total_pnl', 'Total PnL from trades')

# Alpha Signal Metrics
metric_cvd_score = Gauge('btc15m_sig_cvd_score', 'Feature: CVD Score')
metric_ofi_score = Gauge('btc15m_sig_ofi_score', 'Feature: OFI Score')
metric_flow_accel = Gauge('btc15m_sig_flow_accel', 'Feature: Flow Acceleration')
metric_signed_score = Gauge('btc15m_sig_signed_score', 'Total Signed Score')
metric_req_score = Gauge('btc15m_sig_req_score', 'Required Min Score')
metric_sigma_b = Gauge('btc15m_sig_sigma_b', 'Belief volatility sigma_B')
metric_bvol_mult = Gauge('btc15m_sig_bvol_mult', 'Belief-volatility multiplier')

# Execution Counters
metric_trades_executed = Counter('btc15m_trades_executed_total', 'Total number of trades executed')
metric_wins = Counter('btc15m_wins_total', 'Total winning trades')
metric_losses = Counter('btc15m_losses_total', 'Total losing trades')

def start_exporter(port=9090):
    """Start the Prometheus exporter server."""
    try:
        start_http_server(port)
        log.info(f"Prometheus exporter started on port {port}")
    except Exception as e:
        log.error(f"Failed to start Prometheus exporter: {e}")

def update_metrics(state, heartbeat: dict, signal_res=None):
    """Called every cycle to update Prometheus registries."""
    try:
        # Balance
        if "wallet_usdc" in heartbeat and heartbeat["wallet_usdc"] is not None:
            metric_wallet_balance.set(heartbeat["wallet_usdc"])
        if "unclaimed_usdc" in heartbeat and heartbeat["unclaimed_usdc"] is not None:
            metric_unclaimed_winnings.set(heartbeat["unclaimed_usdc"])

        # Exposure
        if state.held_position.side:
            metric_exposure.set(state.held_position.size_usd or 0)
        else:
            metric_exposure.set(0)

        # Performance
        if "perf_db" in heartbeat:
            perf = heartbeat["perf_db"]
            metric_win_rate.set(perf.get("win_rate", 0))
            metric_profit_factor.set(perf.get("profit_factor", 0))
            metric_sharpe.set(perf.get("sharpe_ratio", 0))

        metric_loss_streak.set(state.loss_streak)
        metric_total_pnl.set(state.total_pnl_usd)

        # Alpha Signals
        if signal_res:
            metric_cvd_score.set(signal_res.cvd_score)
            metric_ofi_score.set(signal_res.ofi_score)
            metric_flow_accel.set(signal_res.flow_accel_score)
            metric_signed_score.set(signal_res.signed_score)
            metric_req_score.set(signal_res.min_score)
            metric_sigma_b.set(getattr(signal_res, "sigma_b", 0.0))
            metric_bvol_mult.set(getattr(signal_res, "bvol_multiplier", 0.0))

    except Exception as e:
        log.error(f"Error updating Prometheus metrics: {e}")
