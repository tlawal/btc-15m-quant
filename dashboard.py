import asyncio
import json
import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("dashboard")

app = FastAPI()
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
    try:
        from main import BUILD_VERSION
    except:
        BUILD_VERSION = "unknown"

    result = {
        "build_version": BUILD_VERSION,
        "polygon_rpc_url_set": bool(Config.POLYGON_RPC_URL),
        "polygon_rpc_url_preview": (Config.POLYGON_RPC_URL[:40] + "...") if Config.POLYGON_RPC_URL else "NOT SET",
        "usdc_contract": Config.POLYGON_USDC_ADDRESS,
        "private_key_set": bool(Config.POLYMARKET_PRIVATE_KEY),
        "can_trade": engine.pm.can_trade if engine else None,
        "trading_halted": engine.state.trading_halted if engine and engine.state else None,
    }

    # Try live balance fetch
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

    return result

# Shared state reference set by the Engine
engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        log.exception("Failed to render index.html from templates dir %s", _templates_dir)
    return HTMLResponse(
        "<html><body>"
        "<h2>Dashboard template missing</h2>"
        "<p>The API is up. The UI template <code>templates/index.html</code> was not found in the runtime container.</p>"
        "<p>Try visiting <code>/api/metrics</code> instead, or ensure the templates folder is deployed.</p>"
        "</body></html>"
    )


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
    if not engine or not engine.state:
        return JSONResponse({"status": "loading"}, status_code=503)
    
    state = engine.state
    
    # Heartbeat data (mostly for latencies and status)
    hb_path = "/data/heartbeat.json" if os.path.exists("/data") else "heartbeat.json"
    hb = {}
    if os.path.exists(hb_path):
        try:
            with open(hb_path, 'r') as f:
                hb = json.load(f)
        except: pass

    # Prepare latest signals for the UI
    # We can get these from the state's memory variables
    return {
        "balance": hb.get("balance", 0.0),
        "wallet_usdc": hb.get("wallet_usdc"),
        "pm_collateral_usdc": hb.get("pm_collateral_usdc"),
        "pm_available_usdc": hb.get("pm_available_usdc"),
        "pm_allowance_usdc": hb.get("pm_allowance_usdc"),
        "position": state.held_position.side or "FLAT",
        "entry_price": state.held_position.avg_entry_price,
        "latencies": state.latencies,
        "trades_15m": state.trades_this_window,
        "loss_streak": state.loss_streak,
        "halted": state.trading_halted,
        "last_posterior_up": state.last_posterior_up,
        "last_posterior_down": state.last_posterior_down,
        "last_market_slug": state.last_market_slug,
        "recent_trades": [
            {
                "ts": t.ts,
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "outcome": t.outcome
            }
            for t in state.trade_history[-10:]
        ]
    }

async def run_dashboard(engine_instance, port=8000):
    global engine
    engine = engine_instance
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()
