# BTC 15m Quant Engine 🐍

Institutional-grade BTC 15-minute binary outcome engine for Polymarket.
Pure async Python — `asyncio` + `aiohttp` + `py-clob-client` + SQLite.

---

## All Audit Fixes Applied

| # | Bug | Fix |
|---|-----|-----|
| 1 | Direction from `signedScore` instead of posterior | `direction` now set by `posterior_final_up > posterior_final_down` in `logic.py` |
| 2 | Strike fallback = live EMA (zeroes z-score) | Binance 15m open → **Coinbase 15m open** → HL mid. Live price never used. |
| 3 | JS variable shadowing | Non-issue in Python — `strike_source` assigned directly to outer scope |
| 4 | Synthetic CVD (all-or-nothing by tick) | Real CVD from Binance `aggTrades` endpoint in `data_feeds.py` |
| 5 | No ADX filter | TAAPI ADX bulk fetch; ADX < 20 blocks directional entry |
| 6 | OBV divergence with 1-period slope | TAAPI `results=5`; 5-bar slope for bull/bear divergence |
| 7 | Fixed thresholds regardless of ATR regime | Regime-adaptive: Low / Normal / High ATR → own edge + score thresholds |
| 8 | `riskPct = 1.0` at small balance = ruin | Floor at 0.25. Quarter-Kelly with streak de-risk. |

---

## File Map

```
main.py              Async engine loop (15s), entry/exit orchestration
config.py            All constants + env loading
state.py             SQLite async persistence (SQLAlchemy + aiosqlite)
data_feeds.py        Binance REST (klines, aggTrades CVD), Coinbase, Hyperliquid L2
polymarket_client.py py-clob-client wrapper (market discovery, OB, orders)
indicators.py        TAAPI bulk fetch + pure-math helpers (normal_cdf, logit, …)
logic.py             All signal scoring, Bayesian posterior, belief-vol, sizing, exits
utils.py             Logging, Telegram alerts, window helpers
```

---

## Prerequisites

- Python 3.12+
- Git
- Railway CLI: `npm install -g @railway/cli`
- A dedicated Polygon wallet (Metamask → export private key)

---

## Step 1 — Local Setup

```bash
# Unzip / clone the project
cd btc-15m-quant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.example .env
```

---

## Step 2 — Create a Dedicated Trading Wallet

1. Open Metamask → **Add Account** → note the **private key**
   (Account Details → Export Private Key)
2. Add **Polygon Mainnet** to Metamask:
   - RPC: `https://polygon-rpc.com`
   - Chain ID: `137`
   - Symbol: `MATIC`
3. Bridge USDC to Polygon → https://app.across.to or Transak
   - **Minimum recommended: $25 USDC**
   - Also send ~0.5 MATIC for gas (pennies per transaction)
4. Connect wallet to https://polymarket.com, complete KYC if prompted,
   and deposit USDC to approve the contract.

Set your private key in `.env`:
```
POLYMARKET_PRIVATE_KEY=0xYOUR64HEXKEY
```

---

## Step 3 — Get a TAAPI Key

1. Sign up at https://taapi.io (free tier works)
2. Copy your API key to `.env`:
```
TAAPI_KEY=your_key_here
```

---

## Step 4 — Generate Polymarket API Credentials

Run the engine once with only `POLYMARKET_PRIVATE_KEY` and `TAAPI_KEY` set.
`py-clob-client` will auto-create credentials on first run, or you can generate them manually:

```bash
python - <<'EOF'
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYMARKET_HOST
import os
from dotenv import load_dotenv
load_dotenv()

client = ClobClient(
    host=POLYMARKET_HOST,
    key=os.getenv("POLYMARKET_PRIVATE_KEY"),
    chain_id=137,
)
creds = client.create_or_derive_api_creds()
print(f"POLYMARKET_API_KEY={creds.api_key}")
print(f"POLYMARKET_API_SECRET={creds.api_secret}")
print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
EOF
```

Copy the output into `.env`.

---

## Step 5 — Set Up Telegram Alerts (Recommended)

1. Message `@BotFather` on Telegram → `/newbot` → copy the **HTTP Token**
2. Send any message to your new bot
3. Get your chat ID:
```bash
curl "https://api.telegram.org/bot<TOKEN>/getUpdates" | python3 -m json.tool | grep '"id"'
```
4. Add to `.env`:
```
TELEGRAM_TOKEN=123456:ABCdef...
TELEGRAM_CHAT_ID=987654321
```

---

## Step 6 — Test Locally

```bash
# Activate venv
source .venv/bin/activate

# Test run (watch first 2 cycles in logs)
python main.py
```

**Expected log output:**
```
2025-01-01T12:00:00  INFO      engine  Engine started. Ensuring Polymarket approvals...
2025-01-01T12:00:01  INFO      engine  Ready. Starting main loop.
2025-01-01T12:00:01  INFO      engine  New 15m window: 1735732800 (12:00:00 UTC)
2025-01-01T12:00:02  INFO      engine  Strike locked: 94250.00 via binance_15m_open
2025-01-01T12:00:04  INFO      engine  signal: score=2.50 dir=DOWN postUp=0.0412 ...
```

**To reset state and start fresh:**
```bash
python main.py --reset
```

---

## Step 7 — Deploy to Railway

### 7a. Create project

```bash
railway login
railway new btc-15m-quant
railway link
```

Or in the Railway web UI:
1. https://railway.com → **New Project** → **Deploy from GitHub**
2. Connect repo → Railway detects the Dockerfile automatically

### 7b. Set environment variables

```bash
railway variables set POLYMARKET_PRIVATE_KEY=0x...
railway variables set POLYMARKET_API_KEY=...
railway variables set POLYMARKET_API_SECRET=...
railway variables set POLYMARKET_API_PASSPHRASE=...
railway variables set TAAPI_KEY=...
railway variables set TELEGRAM_TOKEN=...
railway variables set TELEGRAM_CHAT_ID=...
railway variables set LOG_LEVEL=INFO
railway variables set DATABASE_URL=sqlite+aiosqlite:////data/state.db
```

Or in web UI → your service → **Variables** tab.

### 7c. Attach persistent volume ⚠️ CRITICAL

Without this, the SQLite database is lost on every redeploy.

**Web UI:**
1. Your service → **Volumes** tab
2. **New Volume** → Mount path: `/data` → Size: 1 GB → **Create**

**CLI:**
```bash
railway volume create --mount /data --size 1
```

### 7d. Deploy

```bash
# Push to GitHub (if using GitHub integration)
git add . && git commit -m "initial deploy" && git push

# Or deploy directly via CLI
railway up
```

---

## Step 8 — Monitor

### Logs (real-time)
```bash
railway logs --tail 200
```

Or: Railway web UI → service → **Deployments** → **View Logs**

### Health check
```bash
curl https://YOUR_APP.railway.app/health
# → {"status": "ok"}
```

### Verify it's trading correctly

Look for these log lines after deploy:

✅ **Good:**
```
Strike locked: 94250.00 via binance_15m_open
signal: score=-5.2 dir=DOWN postUp=0.0412 edge=0.0588 gates=CLEAR
ENTRY: NO @ 0.9220 shares=6.00 usd=$5.52 score=-5.20 post=0.9588
```

⚠️ **Investigate if you see:**
```
Could not resolve strike price        ← Binance + Coinbase both down
No active BTC 15m market found        ← Polymarket gap between markets
TRADING HALTED: 3 consecutive losses  ← Loss streak hit; reset required
signal: ... gates=['edge_insufficient', 'score_low']  ← Normal; not every cycle trades
```

---

## Step 9 — Resetting After a Loss Halt

If 3 consecutive losses trigger a halt:

**Option A — Railway shell:**
```bash
railway shell
python main.py --reset
```

**Option B — Edit state directly:**
```bash
railway shell
python3 -c "
import sqlite3, json
conn = sqlite3.connect('/data/state.db')
conn.execute(\"INSERT OR REPLACE INTO kv VALUES ('loss_streak', '0')\")
conn.execute(\"INSERT OR REPLACE INTO kv VALUES ('trading_halted', 'false')\")
conn.commit()
conn.close()
print('Done')
"
```

Then restart the service in Railway UI.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Railway Starter plan | $5/month |
| Compute (~0.5 vCPU, 256 MB RAM) | ~$2–5/month |
| Persistent volume (1 GB) | $0.25/month |
| TAAPI free tier | Free (500 req/day) |
| Binance / Coinbase / HL APIs | Free |
| Polymarket trading fee | 2% on winnings only |
| **Total** | **~$8–11/month** |

**TAAPI free tier note:** 500 requests/day. At 1 bulk call per 15s cycle with 96 cycles/day = well within limit. If you run faster cycles, upgrade to TAAPI Pro.

---

## Tuning Parameters

All live-tuneable via Railway env variables (restart required):

| Variable | Default | Effect |
|----------|---------|--------|
| `LOG_LEVEL` | `INFO` | Set `DEBUG` to see every indicator value |
| `MIN_SCORE_*` | 3/4/5 | Higher = more selective entries |
| `REQUIRED_EDGE_*` | 0.025/0.035/0.05 | Higher = only take higher-edge trades |
| `ADX_TREND_THRESHOLD` | 20.0 | Raise to 25 for strict trending-only |
| `RISK_TIER_50` | 0.25 | Reduce to 0.15 for very conservative sizing |
| `STREAK_HALT_AT` | 3 | Raise to 5 if halting too aggressively |
| `TAKE_PROFIT_PRICE` | 0.97 | Raise to 0.98 to hold longer into TP |

---

## Security Notes

- **Never commit `.env`** — add `.env` to `.gitignore` (already included)
- Use a **dedicated wallet** with only your trading capital — never your main wallet
- Start with **$25 USDC** and verify real trades before scaling
- Monitor Telegram alerts actively — silence = something is wrong
