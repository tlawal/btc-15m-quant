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
_TEMPLATES_DIR_CANDIDATES = [
    os.path.join(_HERE, "templates"),
    os.path.join(os.getcwd(), "templates"),
]
_templates_dir = next((p for p in _TEMPLATES_DIR_CANDIDATES if os.path.isdir(p)), None)
templates = Jinja2Templates(directory=_templates_dir or "templates")

# Shared state reference set by the Engine
engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if _templates_dir:
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
