import asyncio
import json
import os
import time
import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("dashboard")
start_time = time.time()

app = FastAPI()

# ── Phase 5: WebSocket connection manager ────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = ConnectionManager()
_HERE = os.path.dirname(os.path.abspath(__file__))
_templates_dir = os.path.join(_HERE, "templates")
templates = Jinja2Templates(directory=_templates_dir)

try:
    log.info(
        "Dashboard templates: dir=%s exists=%s files=%s",
        _templates_dir,
        os.path.isdir(_templates_dir),
        os.listdir(_templates_dir) if os.path.isdir(_templates_dir) else None,
    )
except Exception:
    log.exception("Failed to inspect templates directory")


@app.get("/health")
async def health():
    from main import BUILD_VERSION
    return {"status": "ok", "build": BUILD_VERSION}

@app.get("/api/debug")
async def debug_balance():
    """Live debug: check balance in real-time and show env config."""
    from config import Config
    from eth_account import Account
    try:
        from main import BUILD_VERSION
    except:
        BUILD_VERSION = "unknown"

    wallet_addr = None
    if Config.POLYMARKET_PRIVATE_KEY and len(Config.POLYMARKET_PRIVATE_KEY) > 20:
        try:
            pk = Config.POLYMARKET_PRIVATE_KEY
            if pk.startswith("0x"): pk = pk[2:]
            wallet_addr = Account.from_key(pk).address
        except:
            pass

    result = {
        "build_version": BUILD_VERSION,
        "wallet_address": wallet_addr,
        "polygon_rpc_url_set": bool(Config.POLYGON_RPC_URL),
        "polygon_rpc_url_preview": (Config.POLYGON_RPC_URL[:40] + "...") if Config.POLYGON_RPC_URL else "NOT SET",
        "usdc_contract": Config.POLYGON_USDC_ADDRESS,
        "private_key_set": bool(Config.POLYMARKET_PRIVATE_KEY),
        "can_trade": engine.pm.can_trade if engine and engine.pm else None,
        "trading_halted": engine.state.trading_halted if engine and engine.state else None,
        "last_cycle_error": getattr(engine, "last_cycle_error", None),
        "db_url": Config.DATABASE_URL,
        "db_engine_url": str(engine.state_mgr.engine.url) if engine and engine.state_mgr else "N/A",
        "is_fallback": "state.db" in Config.DATABASE_URL and ("data" not in Config.DATABASE_URL),
    }

    if engine and engine.pm:
        try:
            wallet = await engine.pm.get_wallet_usdc_balance()
            result["live_wallet_usdc"] = wallet
        except Exception as e:
            result["live_wallet_usdc_error"] = str(e)
        try:
            margin = await engine.pm.get_margin()
            result["live_margin"] = margin
        except Exception as e:
            result["live_margin_error"] = str(e)

    # Check heartbeat file
    hb_path = "/data/heartbeat.json" if os.path.isdir("/data") else "heartbeat.json"
    result["hb_path"] = hb_path
    result["hb_exists"] = os.path.exists(hb_path)
    if result["hb_exists"]:
        result["hb_mtime"] = os.path.getmtime(hb_path)
        try:
            with open(hb_path, "r") as f:
                content = f.read()
                result["hb_content_len"] = len(content)
                result["hb_content_preview"] = content[:200]
        except Exception as e:
            result["hb_read_error"] = str(e)
            
    # Check general logs if we can
    log_path = "/data/structured_logs.json" if os.path.exists("/data") else "structured_logs.json"
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                result["last_logs"] = lines[-5:]
        except Exception:
            pass

    return result

@app.get("/api/logs")
async def get_logs(limit: int = 240):
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in lines]
    except:
        return {"error": "Log file not found"}

# Shared state reference set by the Engine
engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        log.warning("Templates not found, serving inline fallback UI")
        return HTMLResponse("""
        <html>
        <head>
            <title>BTC Quant - Emergency UI</title>
            <style>
                body { background: #0f172a; color: #f8fafc; font-family: monospace; padding: 20px; }
                .card { background: #1e293b; border-radius: 8px; padding: 20px; border: 1px solid #334155; }
                pre { background: #000; padding: 10px; border-radius: 4px; overflow: auto; }
                .status { color: #10b981; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>📊 BTC Quant - Emergency Dashboard</h2>
                <p>Status: <span id="status" class="status">Loading...</span></p>
                <p>Build: <span id="build">--</span> | DB: <span id="db">--</span></p>
                <hr style="border-color: #334155">
                <h4>Live Metrics</h4>
                <pre id="metrics">Waiting for data...</pre>
                <h4>Recent Logs</h4>
                <pre id="logs">Waiting for logs...</pre>
            </div>
            <script>
                async function update() {
                    try {
                        const r = await fetch('/api/metrics');
                        const data = await r.json();
                        document.getElementById('status').innerText = data.status || 'Running';
                        document.getElementById('build').innerText = data.version || 'v2.2';
                        document.getElementById('db').innerText = data.db_url || 'default';
                        document.getElementById('metrics').innerText = JSON.stringify(data, null, 2);
                    } catch(e) { document.getElementById('status').innerText = 'API Error'; }
                    
                    try {
                        const r2 = await fetch('/api/logs?limit=10');
                        const data2 = await r2.json();
                        document.getElementById('logs').innerText = data2.logs.map(l => `[${l.type}] ${JSON.stringify(l.data)}`).join('\\n');
                    } catch(e) {}
                }
                setInterval(update, 3000);
                update();
            </script>
        </body>
        </html>
        """)

@app.post("/api/kill")
async def kill_switch(request: Request):
    try:
        data = await request.json()
        pwd = data.get("password")
        from config import Config
        if pwd == Config.KILL_SWITCH_PASSWORD:
            Config.KILL_SWITCH = True
            log.warning("KILL SWITCH ACTIVATED VIA DASHBOARD")
            return {"status": "success", "message": "Kill switch activated"}
        else:
            return JSONResponse({"status": "error", "message": "Invalid password"}, status_code=403)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/debug/templates")
async def debug_templates():
    info = {
        "cwd": os.getcwd(),
        "here": _HERE,
        "templates_dir": _templates_dir,
        "templates_dir_exists": os.path.isdir(_templates_dir),
        "index_exists": os.path.exists(os.path.join(_templates_dir, "index.html")),
    }
    try:
        info["templates_dir_files"] = os.listdir(_templates_dir)
    except Exception as e:
        info["templates_dir_files_error"] = str(e)
    return info

@app.get("/api/metrics")
async def get_metrics():
    if not engine:
        return JSONResponse({"status": "starting"}, status_code=503)
        
    if not engine.state or not getattr(engine, "_running", False):
        return JSONResponse({
            "status": "initializing",
            "db_url": Config.DATABASE_URL,
            "can_trade": engine.pm.can_trade
        }, status_code=503)
    
    state = engine.state
    
    # Heartbeat data (mostly for latencies and status)
    hb_path = "/data/heartbeat.json" if os.path.isdir("/data") else "heartbeat.json"
    hb = {}
    if os.path.exists(hb_path):
        try:
            with open(hb_path, 'r') as f:
                hb = json.load(f)
        except: pass
    if not hb and engine and engine.pm:
        hb["wallet_usdc"] = await engine.pm.get_wallet_usdc_balance() or 0.0

    # Prepare latest signals for the UI
    total_trades = state.total_trades
    wins = state.total_wins
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0
    avg_trade = (state.total_pnl_usd / total_trades) if total_trades > 0 else 0.0
    
    balance = hb.get("wallet_usdc") or hb.get("balance", 0.0)
    open_pos = 1 if state.held_position.side else 0
    exposure = state.held_position.size_usd if state.held_position.side else 0.0

    # Base payload on heartbeat to not drop anything (signal, position, etc.)
    metrics = hb.copy()
    
    # Overwrite/inject dynamic runtime metrics + performance
    metrics.update({
        "balance": hb.get("balance", 0.0),
        "status": "Running" if getattr(engine, "_running", False) else "Stopped",
        "uptime_sec": int(time.time() - start_time),
        "version": getattr(engine, "BUILD_VERSION", "v1.0.0"),
        "wallet_usdc": balance,
        "open_positions": open_pos,
        "exposure_usd": exposure,
        "strike_source": getattr(state, "strike_source", "none"),
        "trading_halted": getattr(state, "trading_halted", False),
        "performance_metrics": getattr(state, "performance_metrics", {}),
        "last_tuned_time": getattr(state, "last_tuned_time", 0),
        "open_positions_api": getattr(state, "open_positions_api", []),
        "recent_trades": [
            {
                "ts": t.ts,
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "outcome": t.outcome,
                "slippage": getattr(t, "slippage", None)
            }
            for t in state.trade_history[-10:]
        ]
    })

    # Surface gate + belief-volatility info at the top level for the UI.
    signal = metrics.get("signal") or {}
    skip_gates = signal.get("skip_gates") or []
    metrics["gate_primary"] = skip_gates[0] if skip_gates else "CLEAR"
    metrics["gate_all"] = skip_gates
    metrics["sigma_b"] = signal.get("sigma_b")
    metrics["bvol_multiplier"] = signal.get("bvol_multiplier")
    
    return metrics

# ── Phase 5 #25: WebSocket real-time push ─────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


async def broadcast_cycle_update(data: dict):
    """Called from engine after each cycle to push live data to all WS clients."""
    await ws_manager.broadcast(data)


# ── Phase 5 #26: Per-signal accuracy metrics (7-day rolling) ─────────────────
@app.get("/api/signal-accuracy")
async def signal_accuracy():
    if not engine:
        return JSONResponse({"status": "loading"}, status_code=503)
    accuracies = engine.optimizer.get_signal_accuracies()
    disabled = list(engine.optimizer.get_disabled_signals())
    return {
        "signals": accuracies,
        "disabled_signals": disabled,
        "kelly_multiplier": engine.optimizer.get_kelly_multiplier(),
    }


# ── Phase 5 #27: Trade P&L attribution by signal ────────────────────────────
@app.get("/api/attribution")
async def trade_attribution():
    if not engine or not engine.state:
        return JSONResponse({"status": "loading"}, status_code=503)

    closed = [t for t in engine.state.trade_history if t.outcome in ("WIN", "LOSS")]
    if not closed:
        return {"message": "No closed trades yet", "attributions": {}}

    # Attribution: for each signal, compute avg PnL when signal was positive vs negative
    from collections import defaultdict
    signal_pnl = defaultdict(lambda: {"pos_pnl": [], "neg_pnl": [], "neutral": 0})

    for t in closed:
        feats = getattr(engine.state, "entry_features", {})
        pnl = t.pnl or 0.0
        for sig_name in ["ema_score", "vwap_score", "rsi_score", "macd_score",
                         "cvd_score", "ofi_score", "tob_score", "cvd_velocity_score",
                         "pm_flow_score", "liq_vacuum_score", "oracle_lag_score",
                         "funding_rate_score", "misprice_score"]:
            val = feats.get(sig_name, 0.0)
            if val > 0:
                signal_pnl[sig_name]["pos_pnl"].append(pnl)
            elif val < 0:
                signal_pnl[sig_name]["neg_pnl"].append(pnl)
            else:
                signal_pnl[sig_name]["neutral"] += 1

    result = {}
    for sig, data in signal_pnl.items():
        pos = data["pos_pnl"]
        neg = data["neg_pnl"]
        result[sig] = {
            "pos_avg_pnl": round(sum(pos) / len(pos), 4) if pos else None,
            "pos_count": len(pos),
            "neg_avg_pnl": round(sum(neg) / len(neg), 4) if neg else None,
            "neg_count": len(neg),
            "neutral_count": data["neutral"],
        }
    return {"attributions": result}


# ── Phase 5 #28: Regime performance breakdown ────────────────────────────────
@app.get("/api/regime-performance")
async def regime_performance():
    if not engine:
        return JSONResponse({"status": "loading"}, status_code=503)
    try:
        metrics = await engine.state_mgr.calculate_performance_metrics()
        return {"regimes": metrics.get("regimes", {}), "overall": metrics}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Phase 5 #29: Order fill analytics ────────────────────────────────────────
@app.get("/api/fill-analytics")
async def fill_analytics():
    if not engine or not engine.state:
        return JSONResponse({"status": "loading"}, status_code=503)

    trades = engine.state.trade_history
    if not trades:
        return {"message": "No trades yet"}

    filled = [t for t in trades if t.outcome != "OPEN"]
    slippages = [t.slippage for t in filled if t.slippage is not None]

    result = {
        "total_orders": len(trades),
        "filled": len(filled),
        "fill_rate": round(len(filled) / len(trades), 3) if trades else 0,
        "avg_slippage_pct": round(sum(slippages) / len(slippages) * 100, 4) if slippages else None,
        "max_slippage_pct": round(max(slippages) * 100, 4) if slippages else None,
        "min_slippage_pct": round(min(slippages) * 100, 4) if slippages else None,
    }
    return result


async def run_dashboard(engine_instance, port=8000):
    global engine
    engine = engine_instance
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", log_config=None)
    server = uvicorn.Server(config)
    
    # We must run `serve` as a task, but await it so it occupies this background task
    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
