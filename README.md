# BTC 15m Quant Engine 🐍

Institutional-grade BTC 15-minute binary outcome engine for Polymarket. 
Pure async Python — `asyncio` + `aiohttp` + `py-clob-client` + PostgreSQL/SQLite.

---

## 🏛️ Phase 5: Institutional Monitoring Stack

The bot now features a full-service monitoring and self-learning stack:

| Feature | Description |
|---|---|
| **Live Dashboard** | Real-time FastAPI dashboard on port 8000 with per-window performance metrics. |
| **Prometheus Exporter** | Time-series signal history and system health metrics for Grafana. |
| **Mid-Window Exit** | Active risk management: exits on 15% SL, 0.6*ATR distance, or momentum reversal. |
| **Self-Learning Layer** | Feature logging for ML training and automated threshold adjustment via `StrategyOptimizer`. |
| **PostgreSQL Support** | Scalable state management for high-frequency trading sessions. |
| **CVD WebSocket** | Real-time 40ms Binance `aggTrades` CVD integration. |

---

## 🛠️ Tech Stack & File Map

- **main.py**: Async engine loop (15s), orchestration of entries/exits/monitoring.
- **polymarket_client.py**: CLOB wrapper with autonomous mid-window exit monitor.
- **signals.py**: Feature calculation (Bayesian posterior, belief-vol, MFI/OFI/CVD).
- **optimizer.py**: Automated threshold tuning based on rolling win-rate decay.
- **inference.py**: Phase 5 ML model inference and signal boosting.
- **dashboard.py**: FastAPI web interface + state debug endpoints.
- **metrics_exporter.py**: Prometheus metrics for institutional monitoring.
- **state.py**: Async database manager (SQLAlchemy + aiosqlite/psycopg).

---

## 🚀 Deployment (Railway)

### 1. Variables
Set the following in Railway:
- `POLYMARKET_PRIVATE_KEY`
- `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`
- *(optional, dev only)* `TAAPI_KEY` for the standalone `test_taapi.py` script — the live engine uses only local indicators.
- `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
- `DATABASE_URL` (Use `sqlite+aiosqlite:////data/state.db` or a Postgres URL)

### 2. Persistent Storage
Ensure a volume is mounted at `/data` to persist `state.db`, `heartbeat.json`, and structured logs.

---

## 🛡️ Risk Management (Fixed & New)

1. **Mid-Window Exit**: The bot continuously monitors open positions. It will exit before window expiry if:
   - `SignedScore` reverses significantly (>-3 for YES, >3 for NO).
   - Unrealized loss exceeds **15%**.
   - Price drifts **>0.6 * ATR** away from strike.
   - Microstructure (CVD/OFI) flips decisively against the position.
2. **Loss Streak Halt**: Automatically stops trading after 3 consecutive losses.
3. **Streak De-risking**: Size is halved if balance is low or on a 2-loss streak.
4. **Adaptive Thresholds**: Required edge and min score scale automatically based on 15m ATR regime.

---

## 📊 Monitoring

- **Dashboard**: `https://your-app.railway.app/`
- **Metrics**: `https://your-app.railway.app/api/metrics`
- **Prometheus**: `https://your-app.railway.app/metrics`
- **Debug**: `https://your-app.railway.app/api/debug` (includes latest cycle errors and heartbeat status)

---

## 📜 Development & Reset

To reset the bot's state (e.g., after a halt):
```bash
python main.py --reset
```
Logs are saved in JSON format as `logs.<timestamp>.json` and `structured_logs.json` for ML training.
