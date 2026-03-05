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
templates = Jinja2Templates(directory="templates")

# Shared state reference set by the Engine
engine = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
