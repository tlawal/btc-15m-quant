import asyncio
import json
import os
import time
import logging
from typing import Optional
from decimal import Decimal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from web3 import AsyncWeb3
import httpx

log = logging.getLogger("dashboard")
start_time = time.time()

app = FastAPI()

# ── Static files (PWA manifest, service worker, icons) ────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

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

# ── Event system for notifications ────────────────────────────────────────────
import collections
from enum import Enum

class EventType(str, Enum):
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    REDEMPTION = "redemption"
    SYSTEM_HALT = "system_halt"
    KILL_SWITCH = "kill_switch"
    TRADING_RESUMED = "trading_resumed"
    ERROR = "error"

_event_buffer: collections.deque = collections.deque(maxlen=50)

# ── VAPID / Web Push ──────────────────────────────────────────────────────────
import base64 as _b64

_HERE = os.path.dirname(os.path.abspath(__file__))
_VAPID_KEY_FILE = os.path.join(_HERE, "vapid_keys.json")
_VAPID_CLAIMS = {"sub": "mailto:admin@btcquant.local"}
_push_subscriptions: dict = {}  # endpoint → subscription dict

def _load_or_create_vapid_keys() -> tuple[str, str]:
    priv_env = os.environ.get("VAPID_PRIVATE_KEY", "")
    pub_env  = os.environ.get("VAPID_PUBLIC_KEY", "")
    if priv_env and pub_env:
        return priv_env, pub_env
    if os.path.exists(_VAPID_KEY_FILE):
        with open(_VAPID_KEY_FILE) as f:
            d = json.load(f)
        return d["private"], d["public"]
    try:
        from pywebpush import Vapid
        v = Vapid()
        v.generate_keys()
        priv = _b64.urlsafe_b64encode(v.private_key.private_bytes_raw()).rstrip(b'=').decode()
        pub  = _b64.urlsafe_b64encode(v.public_key.public_bytes_raw()).rstrip(b'=').decode()
        with open(_VAPID_KEY_FILE, "w") as f:
            json.dump({"private": priv, "public": pub}, f)
        log.info("VAPID keys generated → %s", _VAPID_KEY_FILE)
        return priv, pub
    except Exception:
        log.warning("pywebpush not available — Web Push disabled")
        return "", ""

_VAPID_PRIVATE, _VAPID_PUBLIC = _load_or_create_vapid_keys()

async def _send_web_push(evt: dict):
    if not _VAPID_PRIVATE or not _push_subscriptions:
        return
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        return
    payload = json.dumps({"title": f"BTC Quant: {evt['event_type'].upper()}", "body": evt["message"], "tag": evt["event_type"]})
    dead = []
    for endpoint, sub in list(_push_subscriptions.items()):
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=sub: webpush(subscription_info=s, data=payload,
                                      vapid_private_key=_VAPID_PRIVATE, vapid_claims=_VAPID_CLAIMS)
            )
        except Exception as ex:
            resp = getattr(ex, "response", None)
            if resp is not None and getattr(resp, "status_code", 0) in (404, 410):
                dead.append(endpoint)
    for ep in dead:
        _push_subscriptions.pop(ep, None)

async def emit_event(event_type: EventType, message: str, data: dict = None):
    """Emit a typed event to all WS clients and buffer for late joiners."""
    evt = {
        "type": "event",
        "event_type": event_type.value,
        "message": message,
        "ts": time.time(),
        "data": data or {},
    }
    _event_buffer.appendleft(evt)
    await ws_manager.broadcast(evt)
    asyncio.create_task(_send_web_push(evt))
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
        "trading_halted_reason": getattr(engine.state, "trading_halted_reason", None) if engine and engine.state else None,
        "session_start_balance": getattr(engine.state, "session_start_balance", None) if engine and engine.state else None,
        "daily_loss_limit_usd": getattr(engine.state, "daily_loss_limit_usd", None) if engine and engine.state else None,
        "current_drawdown_usd": getattr(engine.state, "current_drawdown_usd", None) if engine and engine.state else None,
        "daily_loss_soft_scale": getattr(engine.state, "daily_loss_soft_scale", None) if engine and engine.state else None,
        "last_entry_telemetry": getattr(engine.state, "last_entry_telemetry", None) if engine and engine.state else None,
        "kill_switch_active": Config.KILL_SWITCH,
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


@app.get("/api/tx/{tx_hash}")
async def get_tx_receipt(tx_hash: str):
    """Fetch a Polygon tx receipt for a given hash (debug/forensics).

    This is intended for investigating missed fills / reconciliation issues by letting
    us confirm on-chain events for a specific trade.
    """
    from config import Config
    if not Config.POLYGON_RPC_URL:
        return JSONResponse({"error": "POLYGON_RPC_URL is not set"}, status_code=400)

    # Basic validation
    if not isinstance(tx_hash, str) or not tx_hash.startswith("0x") or len(tx_hash) != 66:
        return JSONResponse({"error": "invalid tx hash"}, status_code=400)

    try:
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
        receipt = await w3.eth.get_transaction_receipt(tx_hash)
        if not receipt:
            return JSONResponse({"error": "tx not found"}, status_code=404)

        # Convert AttributeDict / bytes into JSON-friendly structures.
        def _hex_or_val(v):
            try:
                if isinstance(v, (bytes, bytearray)):
                    return "0x" + bytes(v).hex()
            except Exception:
                pass
            return v

        out = {}
        for k, v in dict(receipt).items():
            if k == "logs" and isinstance(v, list):
                logs = []
                for lg in v:
                    d = dict(lg)
                    d["data"] = _hex_or_val(d.get("data"))
                    d["topics"] = [_hex_or_val(t) for t in (d.get("topics") or [])]
                    logs.append(d)
                out["logs"] = logs
            else:
                out[k] = _hex_or_val(v)
        return out
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/signal-history")
async def get_signal_history(limit: int = 240):
    """Return last N structured log entries for Plotly charts."""
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-limit:]
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # structured_logs format: {"ts":..., "type":"signal", "data":{...}}
            if entry.get("type") != "signal":
                continue
            sig = entry.get("data") or {}
            out.append({
                "ts":               entry.get("ts", 0),
                "signed_score":     sig.get("signed_score"),
                "posterior_final_up": sig.get("posterior_final_up"),
                "cvd":              sig.get("cvd_score"),
                "ofi":              sig.get("ofi_score"),
                "regime":           sig.get("regime"),
            })
        return out
    except Exception:
        return []

@app.get("/api/review")
async def get_review():
    """Return the latest nightly trade journal markdown."""
    data_dir = "/data" if os.path.isdir("/data") else "."
    try:
        # Find latest nightly_review_*.md
        files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("nightly_review_") and f.endswith(".md")],
            reverse=True
        )
        if not files:
            return {"date": None, "content": None, "error": "No review available yet"}
        latest = files[0]
        date_str = latest.replace("nightly_review_", "").replace(".md", "")
        with open(os.path.join(data_dir, latest), "r") as f:
            content = f.read()
        return {"date": date_str, "content": content}
    except Exception as e:
        return {"date": None, "content": None, "error": str(e)}

@app.post("/api/resume")
async def resume_trading(password: str = ""):
    """Manual override: clear trading_halted and reset loss_streak."""
    from config import Config
    if password != Config.KILL_SWITCH_PASSWORD:
        return JSONResponse({"ok": False, "error": "bad password"}, status_code=403)
    if not engine or not engine.state:
        return JSONResponse({"ok": False, "error": "engine not ready"}, status_code=503)
    engine.state.trading_halted = False
    engine.state.loss_streak = 0
    Config.KILL_SWITCH = False  # also clear in-memory kill switch
    await engine.state_mgr.save(engine.state)
    log.info("MANUAL RESUME: trading_halted + KILL_SWITCH cleared via /api/resume")
    return {"ok": True, "message": "Trading resumed — halted=False, loss_streak=0, kill_switch=False"}

@app.post("/api/logs/clear")
async def clear_logs():
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    open(log_path, "w").close()
    return {"status": "cleared", "path": log_path}

@app.get("/api/logs/download")
async def download_structured_logs():
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}
    return FileResponse(
        path=log_path,
        media_type="application/octet-stream",
        filename=os.path.basename(log_path),
    )

@app.get("/api/logs/range")
async def get_logs_range(start_ts: int, end_ts: int, types: str = "", limit: int = 5000):
    """Return structured log entries between [start_ts, end_ts).

    This avoids pulling huge log payloads over the network when doing forensic
    reconstruction.
    """
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}

    want_types = set(t.strip() for t in (types or "").split(",") if t.strip())
    out = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                ts = entry.get("ts")
                if ts is None:
                    continue
                try:
                    ts_i = int(ts)
                except Exception:
                    continue
                if ts_i < int(start_ts) or ts_i >= int(end_ts):
                    continue
                if want_types and entry.get("type") not in want_types:
                    continue
                out.append(entry)
                if len(out) >= int(limit):
                    break
        return out
    except Exception:
        return {"error": "Log file not found"}


@app.get("/api/text-logs/list")
async def list_text_logs(prefix: str = "logs."):
    """List files in /data that look like logs or forensic artifacts."""
    base_dir = "/data" if os.path.isdir("/data") else "."
    try:
        files = []
        for name in os.listdir(base_dir):
            if prefix and not name.startswith(prefix):
                continue
            if not (
                name.endswith(".log")
                or name.startswith("logs.")
                or name.startswith("structured_logs.json")
            ):
                continue
            path = os.path.join(base_dir, name)
            if os.path.isfile(path):
                try:
                    st = os.stat(path)
                    files.append({"name": name, "size": int(st.st_size), "mtime": float(st.st_mtime)})
                except Exception:
                    files.append({"name": name})
        files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return {"base_dir": base_dir, "files": files}
    except Exception as e:
        return {"error": str(e), "base_dir": base_dir}


@app.get("/api/text-logs/download")
async def download_text_log(name: str):
    """Download a specific /data log/artifact file by name."""
    base_dir = "/data" if os.path.isdir("/data") else "."
    safe = os.path.basename(name)
    if safe != name:
        return {"error": "Invalid filename"}
    if not (
        safe.endswith(".log")
        or safe.startswith("logs.")
        or safe.startswith("structured_logs.json")
    ):
        return {"error": "File not allowed"}
    path = os.path.join(base_dir, safe)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=safe,
    )

@app.get("/api/logs")
async def get_logs(limit: int = 240):
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line.strip()) for line in lines if line.strip()]
    except:
        return {"error": "Log file not found"}

@app.post("/api/db/reset")
async def reset_db():
    """Reset all performance metrics and state"""
    db_path = "/data/state.db" if os.path.isdir("/data") else "state.db"
    removed = False
    if os.path.exists(db_path):
        os.remove(db_path)
        removed = True
    return {"status": "reset", "path": db_path, "removed": removed}

@app.get("/api/trade-detail")
async def trade_detail(market_slug: str = "", request: Request = None):
    """Forensic audit endpoint: full trade detail by market_slug, no DB download needed.
    Returns partial_exits with reasons, entry features/indicators, entry telemetry, and PnL."""
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "loading"}, status_code=503)
    if not market_slug:
        return JSONResponse({"error": "market_slug query param required"}, status_code=400)

    # Search in-memory trade_history
    trade = None
    for tr in reversed(engine.state.trade_history):
        _slug = getattr(engine.state, "last_market_slug", "") or ""
        _win = str(getattr(tr, "window", ""))
        if _slug == market_slug or market_slug.endswith(_win):
            trade = tr
            break
    if trade is None:
        # Fallback: search closed_trades in DB
        try:
            rows = await engine.state_mgr.fetch_closed_trades(market_slug=market_slug)
            if rows:
                return {"source": "closed_trades_db", "trade": rows[0]}
        except Exception:
            pass
        return JSONResponse({"error": f"Trade not found for {market_slug}"}, status_code=404)

    result = {
        "source": "trade_history",
        "market_slug": market_slug,
        "ts": getattr(trade, "ts", None),
        "side": getattr(trade, "side", None),
        "entry_price": getattr(trade, "entry_price", None),
        "exit_price": getattr(trade, "exit_price", None),
        "size": getattr(trade, "size", None),
        "pnl": getattr(trade, "pnl", None),
        "outcome": getattr(trade, "outcome", None),
        "exit_reason": getattr(trade, "exit_reason", None),
        "score": getattr(trade, "score", None),
        "slippage": getattr(trade, "slippage", None),
        "tx_hash": getattr(trade, "tx_hash", None),
        "window": getattr(trade, "window", None),
        "partial_exits": list(getattr(trade, "partial_exits", []) or []),
    }

    # Add entry features from closed_trades DB (has the full features JSON)
    try:
        rows = await engine.state_mgr.fetch_closed_trades(market_slug=market_slug)
        if rows and rows[0].get("features"):
            result["features"] = rows[0]["features"]
    except Exception:
        pass

    # Add entry telemetry from state (if available)
    _tel = getattr(engine.state, "last_entry_telemetry", None)
    if _tel and _tel.get("market_slug") == market_slug:
        result["entry_telemetry"] = _tel

    return result

@app.get("/api/db/download")
async def download_db():
    db_path = "/data/state.db" if os.path.isdir("/data") else "state.db"
    if not os.path.exists(db_path):
        return JSONResponse({"error": "db_not_found", "path": db_path}, status_code=404)
    return FileResponse(
        db_path,
        media_type="application/octet-stream",
        filename=os.path.basename(db_path),
    )

@app.post("/api/purge-all")
async def purge_all(request: Request):
    """Purge ALL persistent state: DB, exit logs, optimizer model/features, calibration, structured logs.
    After calling this, restart the service to reinitialize in-memory state."""
    deny = _require_admin(request)
    if deny:
        return deny
    base = "/data" if os.path.isdir("/data") else "."
    files_to_purge = [
        "state.db",
        "exit_outcomes.jsonl",
        "trade_features.jsonl",
        "calibration_log.jsonl",
        "optimizer_model.joblib",
        "platt_scaler.json",
        "structured_logs.json",
        "outcomes.jsonl",
    ]
    purged = []
    for fname in files_to_purge:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                purged.append(fname)
            except Exception as e:
                purged.append(f"{fname} (error: {e})")
        else:
            purged.append(f"{fname} (not found)")
    return {"status": "purged", "purged": purged, "note": "Restart service to reinitialize in-memory state"}

@app.get("/api/logs")
async def get_logs(limit: int = 240):
    log_path = "/data/structured_logs.json" if os.path.isdir("/data") else "structured_logs.json"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line.strip()) for line in lines if line.strip()]
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
        if pwd != Config.KILL_SWITCH_PASSWORD:
            return JSONResponse({"status": "error", "message": "Invalid password"}, status_code=403)
        Config.KILL_SWITCH = True
        # Also set trading_halted so the UI halt banner appears immediately
        if engine and engine.state:
            engine.state.trading_halted = True
            await engine.state_mgr.save(engine.state)
        log.warning("KILL SWITCH ACTIVATED VIA DASHBOARD")
        try:
            await emit_event(EventType.KILL_SWITCH, "Kill switch activated via dashboard")
        except Exception:
            pass
        return {"status": "success", "message": "Kill switch activated — trading halted"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def _require_admin(request: Request) -> Optional[JSONResponse]:
    try:
        from config import Config
        token = (Config.DASHBOARD_ADMIN_TOKEN or "").strip()
        if not token:
            return JSONResponse(
                {"status": "error", "message": "Admin token not configured"},
                status_code=503,
            )
        got = (request.headers.get("x-admin-token") or "").strip()
        if got != token:
            return JSONResponse(
                {"status": "error", "message": "Unauthorized"},
                status_code=403,
            )
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "Unauthorized"},
            status_code=403,
        )
    return None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _serialize_trade_record(tr) -> dict:
    return {
        "ts": getattr(tr, "ts", None),
        "side": getattr(tr, "side", None),
        "entry_price": getattr(tr, "entry_price", None),
        "exit_price": getattr(tr, "exit_price", None),
        "pnl": getattr(tr, "pnl", None),
        "outcome": getattr(tr, "outcome", None),
        "slippage": getattr(tr, "slippage", None),
        "size": getattr(tr, "size", None),
        "tx_hash": getattr(tr, "tx_hash", None),
        "exit_reason": getattr(tr, "exit_reason", None),
        "partial_exits": list(getattr(tr, "partial_exits", []) or []),
    }


def _trade_pnl_usd(tr) -> float:
    pnl = getattr(tr, "pnl", None)
    if pnl is None:
        return 0.0
    return _safe_float(pnl) * _safe_float(getattr(tr, "entry_price", 0.0)) * _safe_float(getattr(tr, "size", 0.0))


def _summarize_trade_history(trades) -> dict:
    closed = [t for t in trades if getattr(t, "outcome", None) in ("WIN", "LOSS")]
    loss_streak = 0
    for tr in reversed(closed):
        if getattr(tr, "outcome", None) == "LOSS":
            loss_streak += 1
        else:
            break
    return {
        "total_trades": len(closed),
        "total_wins": sum(1 for t in closed if getattr(t, "outcome", None) == "WIN"),
        "total_losses": sum(1 for t in closed if getattr(t, "outcome", None) == "LOSS"),
        "total_pnl_usd": round(sum(_trade_pnl_usd(t) for t in closed), 6),
        "loss_streak": loss_streak,
    }


def _loads_json_array(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            val = json.loads(raw)
            return val if isinstance(val, list) else []
        except Exception:
            return []
    return []


def _effective_activity_price(row: dict) -> float:
    size = _safe_float((row or {}).get("size"))
    usdc_size = _safe_float((row or {}).get("usdcSize"))
    if size > 0 and usdc_size > 0:
        return usdc_size / size
    return _safe_float((row or {}).get("price"))


def _winning_outcome_from_event(event: dict) -> Optional[str]:
    markets = (event or {}).get("markets") or []
    if not markets:
        return None
    outcomes = _loads_json_array(markets[0].get("outcomes"))
    prices = _loads_json_array(markets[0].get("outcomePrices"))
    for idx, raw_price in enumerate(prices):
        try:
            if float(raw_price) >= 0.999:
                return outcomes[idx] if idx < len(outcomes) else None
        except Exception:
            continue
    return None


def _side_matches_outcome(side: str, outcome_label: Optional[str]) -> bool:
    label = str(outcome_label or "").strip().lower()
    side = str(side or "").upper()
    if side == "YES":
        return ("yes" in label) or ("up" in label)
    if side == "NO":
        return ("no" in label) or ("down" in label)
    return False


async def _fetch_trade_repair_inputs(wallet: str, market_slug: str) -> dict:
    activity_url = f"https://data-api.polymarket.com/activity?user={wallet.lower()}"
    event_url = f"https://gamma-api.polymarket.com/events/slug/{market_slug}"

    async with httpx.AsyncClient(timeout=20) as client:
        activity_resp, event_resp = await asyncio.gather(
            client.get(activity_url),
            client.get(event_url),
        )

    activity_resp.raise_for_status()
    event_resp.raise_for_status()
    activity = activity_resp.json() or []
    event = event_resp.json() or {}

    rows = sorted(
        [
            r for r in activity
            if r.get("slug") == market_slug and r.get("type") == "TRADE"
        ],
        key=lambda r: int(r.get("timestamp") or 0),
    )
    buys = [r for r in rows if str(r.get("side") or "").upper() == "BUY"]
    if not buys:
        raise ValueError(f"No BUY activity found for {market_slug}")

    buy = buys[-1]
    buy_asset = str(buy.get("asset") or "")
    sells = [
        r for r in rows
        if str(r.get("side") or "").upper() == "SELL"
        and int(r.get("timestamp") or 0) >= int(buy.get("timestamp") or 0)
        and (not buy_asset or str(r.get("asset") or "") == buy_asset)
    ]
    if not sells:
        raise ValueError(f"No SELL activity found for {market_slug}")

    winner = _winning_outcome_from_event(event)
    if not winner:
        raise ValueError(f"Could not determine winning outcome for {market_slug}")

    sell_size = sum(_safe_float(r.get("size")) for r in sells)
    sell_usdc = sum(_safe_float(r.get("usdcSize")) for r in sells)
    if sell_size <= 0 or sell_usdc <= 0:
        raise ValueError(f"Matched SELL activity for {market_slug} has zero size/notional")

    return {
        "buy": buy,
        "sells": sells,
        "event": event,
        "winning_outcome": winner,
        "buy_size": _safe_float(buy.get("size")),
        "buy_usdc": _safe_float(buy.get("usdcSize")),
        "buy_effective_price": _effective_activity_price(buy),
        "sell_size": sell_size,
        "sell_usdc": sell_usdc,
        "sell_effective_price": sell_usdc / sell_size,
        "sell_ts": max(int(r.get("timestamp") or 0) for r in sells),
    }


def _build_repaired_trade(tr, repair_inputs: dict) -> dict:
    buy_size = repair_inputs["buy_size"]
    buy_usdc = repair_inputs["buy_usdc"]
    sell_size = repair_inputs["sell_size"]
    sell_usdc = repair_inputs["sell_usdc"]
    side_won = _side_matches_outcome(getattr(tr, "side", None), repair_inputs["winning_outcome"])
    remaining_size = max(0.0, buy_size - sell_size)
    settle_px = 1.0 if side_won else 0.0
    settle_recovered = remaining_size * settle_px
    total_recovered = sell_usdc + settle_recovered
    blended_exit_price = total_recovered / buy_size if buy_size > 0 else 0.0
    blended_pnl = (total_recovered / buy_usdc - 1.0) if buy_usdc > 0 else 0.0
    partial_reason = (
        ((getattr(tr, "partial_exits", None) or [{}])[0] or {}).get("reason")
        or "TP_FULL"
    )
    return {
        "size": buy_size,
        "entry_price": repair_inputs["buy_effective_price"],
        "exit_price": blended_exit_price,
        "pnl": blended_pnl,
        "outcome": "WIN" if blended_pnl >= 0 else "LOSS",
        "exit_reason": "AUTO_SETTLE_WIN" if blended_pnl >= 0 else "AUTO_SETTLE_LOSS",
        "partial_exits": [
            {
                "size": sell_size,
                "price": repair_inputs["sell_effective_price"],
                "reason": partial_reason,
                "ts": repair_inputs["sell_ts"],
            }
        ],
        "side_won": side_won,
        "remaining_size": remaining_size,
        "settle_price": settle_px,
        "pnl_usd": total_recovered - buy_usdc,
    }


@app.post("/api/sweep-dust")
async def sweep_dust(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    from config import Config
    from eth_account import Account

    if not Config.POLYGON_RPC_URL:
        return JSONResponse({"status": "error", "message": "POLYGON_RPC_URL not set"}, status_code=503)
    if not Config.POLYMARKET_PRIVATE_KEY:
        return JSONResponse({"status": "error", "message": "POLYMARKET_PRIVATE_KEY not set"}, status_code=503)

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    dry_run = bool((body or {}).get("dry_run", True))
    max_transfers = int((body or {}).get("max_transfers", 20) or 20)
    min_shares = Decimal(str((body or {}).get("min_shares", "0") or "0"))

    expected_from = "0x7AbA1F81034d418A4DED1613626cA7573FD85153"
    to_wallet = "0xddb76ec1164a72d01211524a0a056bc9c1d8574c"

    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    from_wallet = account.address
    if from_wallet.lower() != expected_from.lower():
        return JSONResponse(
            {
                "status": "error",
                "message": f"Configured private key address mismatch. expected={expected_from} actual={from_wallet}",
            },
            status_code=400,
        )

    try:
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
        if not await w3.is_connected():
            return JSONResponse({"status": "error", "message": "Could not connect to Polygon RPC"}, status_code=503)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"RPC init failed: {e}"}, status_code=503)

    CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CONDITIONAL_ABI = [
        {
            "inputs": [
                {"internalType": "address", "name": "from", "type": "address"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"},
                {"internalType": "bytes", "name": "data", "type": "bytes"},
            ],
            "name": "safeTransferFrom",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "address", "name": "account", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
            ],
            "name": "balanceOf",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
                {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
                {"internalType": "uint256", "name": "indexSet", "type": "uint256"},
            ],
            "name": "getCollectionId",
            "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "address", "name": "collateralToken", "type": "address"},
                {"internalType": "bytes32", "name": "collectionId", "type": "bytes32"},
            ],
            "name": "getPositionId",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "pure",
            "type": "function",
        },
    ]

    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=CONDITIONAL_ABI)
    parent_collection_id = "0x" + ("00" * 32)
    collateral_tokens = [
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    ]

    async def compute_position_id(condition_id: str, index_set: int) -> tuple[Optional[int], Optional[str]]:
        for collateral in collateral_tokens:
            try:
                collection_id = await conditional.functions.getCollectionId(
                    parent_collection_id,
                    w3.to_bytes(hexstr=condition_id),
                    int(index_set),
                ).call()
                position_id = await conditional.functions.getPositionId(
                    w3.to_checksum_address(collateral),
                    collection_id,
                ).call()
                return int(position_id), collateral
            except Exception:
                continue
        return None, None

    def is_resolved(p: dict) -> bool:
        if p.get("redeemable") is True:
            return True
        if p.get("closed") is True:
            return True
        if p.get("resolved") is True:
            return True
        if p.get("settled") is True:
            return True
        if p.get("status") in ("CLOSED", "RESOLVED", "SETTLED"):
            return True
        return False

    try:
        url = f"https://data-api.polymarket.com/positions?user={from_wallet.lower()}"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            r.raise_for_status()
            positions = r.json() or []
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Data API positions fetch failed: {e}"}, status_code=502)

    resolved = [p for p in positions if is_resolved(p)]
    transfers: list[dict] = []
    broadcasted = 0

    base_nonce: Optional[int] = None
    if not dry_run:
        try:
            base_nonce = await w3.eth.get_transaction_count(from_wallet, "pending")
        except Exception:
            base_nonce = await w3.eth.get_transaction_count(from_wallet)

    for p in resolved:
        condition_id = p.get("conditionId") or p.get("condition_id")
        if not condition_id:
            continue

        for index_set in (1, 2):
            if broadcasted >= max_transfers:
                break

            position_id, collateral = await compute_position_id(condition_id, index_set)
            if not position_id:
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "status": "skipped",
                        "reason": "could_not_compute_position_id",
                    }
                )
                continue

            try:
                bal_raw = await conditional.functions.balanceOf(from_wallet, position_id).call()
            except Exception as e:
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "token_id": str(position_id),
                        "collateral": collateral,
                        "status": "skipped",
                        "reason": f"balance_check_failed: {e}",
                    }
                )
                continue

            if not bal_raw or int(bal_raw) <= 0:
                continue

            shares = Decimal(int(bal_raw)) / Decimal(1_000_000)
            if shares < min_shares:
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "token_id": str(position_id),
                        "collateral": collateral,
                        "shares": float(shares),
                        "status": "skipped",
                        "reason": f"below_min_shares({min_shares})",
                    }
                )
                continue

            if dry_run:
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "token_id": str(position_id),
                        "collateral": collateral,
                        "shares": float(shares),
                        "status": "dry_run",
                    }
                )
                continue

            try:
                if base_nonce is None:
                    base_nonce = await w3.eth.get_transaction_count(from_wallet, "pending")
                nonce = base_nonce
                tx = await conditional.functions.safeTransferFrom(
                    from_wallet,
                    w3.to_checksum_address(to_wallet),
                    int(position_id),
                    int(bal_raw),
                    b"",
                ).build_transaction(
                    {
                        "from": from_wallet,
                        "nonce": nonce,
                        "chainId": int(getattr(Config, "CHAIN_ID", 137) or 137),
                        "gas": 220000,
                        "gasPrice": await w3.eth.gas_price,
                    }
                )
                signed = w3.eth.account.sign_transaction(tx, private_key=Config.POLYMARKET_PRIVATE_KEY)
                raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
                tx_hash = await w3.eth.send_raw_transaction(raw)
                receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
                ok = int(getattr(receipt, "status", 0) or 0) == 1
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "token_id": str(position_id),
                        "collateral": collateral,
                        "shares": float(shares),
                        "status": "sent" if ok else "failed",
                        "tx_hash": tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
                    }
                )
                broadcasted += 1
                base_nonce += 1
            except Exception as e:
                transfers.append(
                    {
                        "condition_id": condition_id,
                        "index_set": index_set,
                        "token_id": str(position_id),
                        "collateral": collateral,
                        "shares": float(shares),
                        "status": "failed",
                        "reason": str(e),
                    }
                )

        if broadcasted >= max_transfers:
            break

    def _hex_topic_addr(addr: str) -> str:
        a = addr.lower().replace("0x", "")
        return "0x" + ("0" * 24) + a

    async def _discover_token_ids_onchain(*, lookback_blocks: int, chunk_size: int) -> tuple[list[str], dict]:
        try:
            latest = await w3.eth.block_number
        except Exception:
            latest = await w3.eth.get_block_number()

        lb = int(lookback_blocks or 0)
        if lb <= 0:
            lb = 2_000_000
        start = max(0, int(latest) - lb)
        end = int(latest)

        cs = int(chunk_size or 0)
        if cs <= 0:
            cs = 200_000

        wallet_topic = _hex_topic_addr(from_wallet)
        sig_single = "0x" + w3.keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex().replace("0x", "")
        sig_batch = "0x" + w3.keccak(text="TransferBatch(address,address,address,uint256[],uint256[])").hex().replace("0x", "")

        token_ids: set[int] = set()

        diagnostics: dict = {
            "source": "onchain_logs",
            "from_block": int(start),
            "to_block": int(end),
            "lookback_blocks": int(lb),
            "chunk_size": int(cs),
            "log_queries": 0,
            "logs_returned": 0,
            "getlogs_errors": [],
        }

        async def fetch_logs_chunked(sig: str, topics: list[Optional[str]]):
            out_logs = []
            cur = int(start)
            while cur <= int(end):
                to_b = min(int(end), cur + cs - 1)
                diagnostics["log_queries"] += 1
                try:
                    chunk = await w3.eth.get_logs(
                        {
                            "fromBlock": cur,
                            "toBlock": to_b,
                            "address": w3.to_checksum_address(CONDITIONAL_TOKENS),
                            "topics": topics,
                        }
                    )
                    if chunk:
                        out_logs.extend(chunk)
                except Exception as e:
                    diagnostics["getlogs_errors"].append(
                        {
                            "fromBlock": int(cur),
                            "toBlock": int(to_b),
                            "topic0": str(sig),
                            "topics": [str(t) if t is not None else None for t in topics],
                            "error": str(e),
                        }
                    )
                cur = to_b + 1
            return out_logs

        logs = []
        logs += await fetch_logs_chunked(sig_single, [sig_single, None, None, wallet_topic])
        logs += await fetch_logs_chunked(sig_single, [sig_single, None, wallet_topic, None])
        logs += await fetch_logs_chunked(sig_batch, [sig_batch, None, None, wallet_topic])
        logs += await fetch_logs_chunked(sig_batch, [sig_batch, None, wallet_topic, None])

        diagnostics["logs_returned"] = int(len(logs))

        for lg in logs:
            try:
                t0 = (lg.get("topics") or [])[0]
                if isinstance(t0, (bytes, bytearray)):
                    t0 = "0x" + bytes(t0).hex()
                data_hex = lg.get("data") or "0x"
                if isinstance(data_hex, (bytes, bytearray)):
                    data = bytes(data_hex)
                else:
                    data = bytes.fromhex(str(data_hex).replace("0x", ""))

                if str(t0).lower() == str(sig_single).lower():
                    if len(data) >= 64:
                        tid = int.from_bytes(data[0:32], "big")
                        token_ids.add(tid)
                elif str(t0).lower() == str(sig_batch).lower():
                    try:
                        from eth_abi import decode

                        ids, _values = decode(["uint256[]", "uint256[]"], data)
                        for tid in ids:
                            try:
                                token_ids.add(int(tid))
                            except Exception:
                                continue
                    except Exception:
                        continue
            except Exception:
                continue

        out: list[str] = []
        for tid in token_ids:
            try:
                bal_raw = await conditional.functions.balanceOf(from_wallet, int(tid)).call()
                if bal_raw and int(bal_raw) > 0:
                    out.append(str(int(tid)))
            except Exception:
                continue

        diagnostics["candidate_token_ids"] = int(len(token_ids))
        diagnostics["held_token_ids"] = int(len(set(out)))
        return list(set(out)), diagnostics

    # Fetch orphaned tokens from Zerion
    orphaned_tokens = []
    orphaned_discovery = {"source": "onchain_logs", "error": "discovery_not_run"}
    try:
        lookback_blocks = int((body or {}).get("lookback_blocks", 2_000_000) or 2_000_000)
        chunk_size = int((body or {}).get("log_chunk", 200_000) or 200_000)
        orphaned_tokens, orphaned_discovery = await _discover_token_ids_onchain(
            lookback_blocks=lookback_blocks,
            chunk_size=chunk_size,
        )
    except Exception as e:
        orphaned_tokens = []
        orphaned_discovery = {"source": "onchain_logs", "error": str(e)}

    # Process orphaned token IDs (tokens not in API)
    orphaned_count = 0
    for token_id_str in orphaned_tokens:
        if broadcasted >= max_transfers:
            break
        try:
            token_id = int(token_id_str)
        except ValueError:
            transfers.append(
                {
                    "token_id": token_id_str,
                    "status": "skipped",
                    "reason": "invalid_token_id",
                }
            )
            continue

        try:
            bal_raw = await conditional.functions.balanceOf(from_wallet, token_id).call()
        except Exception as e:
            transfers.append(
                {
                    "token_id": str(token_id),
                    "status": "skipped",
                    "reason": f"balance_check_failed: {e}",
                }
            )
            continue

        if not bal_raw or int(bal_raw) <= 0:
            continue

        shares = Decimal(int(bal_raw)) / Decimal(1_000_000)
        if shares < min_shares:
            transfers.append(
                {
                    "token_id": str(token_id),
                    "shares": float(shares),
                    "status": "skipped",
                    "reason": f"below_min_shares({min_shares})",
                }
            )
            continue

        if dry_run:
            transfers.append(
                {
                    "token_id": str(token_id),
                    "shares": float(shares),
                    "status": "dry_run",
                }
            )
            orphaned_count += 1
            continue

        try:
            if base_nonce is None:
                base_nonce = await w3.eth.get_transaction_count(from_wallet, "pending")
            nonce = base_nonce
            tx = await conditional.functions.safeTransferFrom(
                from_wallet,
                w3.to_checksum_address(to_wallet),
                int(token_id),
                int(bal_raw),
                b"",
            ).build_transaction(
                {
                    "from": from_wallet,
                    "nonce": nonce,
                    "chainId": int(getattr(Config, "CHAIN_ID", 137) or 137),
                    "gas": 220000,
                    "gasPrice": await w3.eth.gas_price,
                }
            )
            signed = w3.eth.account.sign_transaction(tx, private_key=Config.POLYMARKET_PRIVATE_KEY)
            raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
            tx_hash = await w3.eth.send_raw_transaction(raw)
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            ok = int(getattr(receipt, "status", 0) or 0) == 1
            transfers.append(
                {
                    "token_id": str(token_id),
                    "shares": float(shares),
                    "status": "sent" if ok else "failed",
                    "tx_hash": tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
                }
            )
            broadcasted += 1
            orphaned_count += 1
            base_nonce += 1
        except Exception as e:
            transfers.append(
                {
                    "token_id": str(token_id),
                    "shares": float(shares),
                    "status": "failed",
                    "reason": str(e),
                }
            )

    return {
        "status": "ok",
        "dry_run": dry_run,
        "from": from_wallet,
        "to": to_wallet,
        "resolved_positions": len(resolved),
        "orphaned_tokens_fetched": len(orphaned_tokens),
        "orphaned_tokens_processed": orphaned_count,
        "orphaned_discovery": orphaned_discovery,
        "transfers": transfers,
    }


@app.post("/api/transfer-conditional-token")
async def transfer_conditional_token(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    from config import Config
    from eth_account import Account

    if not Config.POLYGON_RPC_URL:
        return JSONResponse({"status": "error", "message": "POLYGON_RPC_URL not set"}, status_code=503)
    if not Config.POLYMARKET_PRIVATE_KEY:
        return JSONResponse({"status": "error", "message": "POLYMARKET_PRIVATE_KEY not set"}, status_code=503)

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    condition_id = str((body or {}).get("condition_id") or (body or {}).get("conditionId") or "").strip()
    if not condition_id:
        return JSONResponse({"status": "error", "message": "condition_id is required"}, status_code=400)
    if not condition_id.startswith("0x"):
        condition_id = "0x" + condition_id
    if len(condition_id) != 66:
        return JSONResponse({"status": "error", "message": "condition_id must be 32-byte hex (0x + 64 hex chars)"}, status_code=400)

    # Convention: index_set=1 or 2; for binary markets we treat 1 as NO, 2 as YES.
    index_set = int((body or {}).get("index_set", 1) or 1)
    if index_set not in (1, 2):
        return JSONResponse({"status": "error", "message": "index_set must be 1 (NO) or 2 (YES)"}, status_code=400)

    dry_run = bool((body or {}).get("dry_run", True))
    max_gas = int((body or {}).get("gas", 220000) or 220000)

    expected_from = "0x7AbA1F81034d418A4DED1613626cA7573FD85153"
    default_to = "0xddb76ec1164a72d01211524a0a056bc9c1d8574c"
    to_wallet = str((body or {}).get("to") or (body or {}).get("to_wallet") or default_to).strip()

    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    from_wallet = account.address
    if from_wallet.lower() != expected_from.lower():
        return JSONResponse(
            {
                "status": "error",
                "message": f"Configured private key address mismatch. expected={expected_from} actual={from_wallet}",
            },
            status_code=400,
        )

    # Safety: only allow sending to the proxy by default.
    if to_wallet.lower() != default_to.lower():
        return JSONResponse(
            {
                "status": "error",
                "message": f"Refusing to transfer to non-proxy address. expected_to={default_to} got_to={to_wallet}",
            },
            status_code=400,
        )

    try:
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
        if not await w3.is_connected():
            return JSONResponse({"status": "error", "message": "Could not connect to Polygon RPC"}, status_code=503)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"RPC init failed: {e}"}, status_code=503)

    CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CONDITIONAL_ABI = [
        {
            "inputs": [
                {"internalType": "address", "name": "from", "type": "address"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"},
                {"internalType": "bytes", "name": "data", "type": "bytes"},
            ],
            "name": "safeTransferFrom",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "address", "name": "account", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
            ],
            "name": "balanceOf",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
                {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
                {"internalType": "uint256", "name": "indexSet", "type": "uint256"},
            ],
            "name": "getCollectionId",
            "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
            "stateMutability": "view",
            "type": "function",
        },
        {
            "inputs": [
                {"internalType": "address", "name": "collateralToken", "type": "address"},
                {"internalType": "bytes32", "name": "collectionId", "type": "bytes32"},
            ],
            "name": "getPositionId",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "pure",
            "type": "function",
        },
    ]

    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=CONDITIONAL_ABI)
    parent_collection_id = "0x" + ("00" * 32)
    collateral_tokens = [
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC.e (bridged)
        "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # USDC (native)
    ]

    position_id = None
    collateral_used = None
    for collateral in collateral_tokens:
        try:
            collection_id = await conditional.functions.getCollectionId(
                parent_collection_id,
                w3.to_bytes(hexstr=condition_id),
                int(index_set),
            ).call()
            pid = await conditional.functions.getPositionId(
                w3.to_checksum_address(collateral),
                collection_id,
            ).call()
            position_id = int(pid)
            collateral_used = collateral
            break
        except Exception:
            continue

    if not position_id:
        return JSONResponse(
            {
                "status": "error",
                "message": "Could not compute position_id from condition_id",
                "condition_id": condition_id,
                "index_set": int(index_set),
            },
            status_code=400,
        )

    try:
        bal_raw = await conditional.functions.balanceOf(from_wallet, int(position_id)).call()
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "message": f"balanceOf failed: {e}",
                "token_id": str(position_id),
            },
            status_code=502,
        )

    if not bal_raw or int(bal_raw) <= 0:
        return {
            "status": "ok",
            "dry_run": dry_run,
            "from": from_wallet,
            "to": to_wallet,
            "condition_id": condition_id,
            "index_set": int(index_set),
            "collateral": collateral_used,
            "token_id": str(position_id),
            "raw_balance": int(bal_raw or 0),
            "message": "No balance to transfer",
        }

    shares = Decimal(int(bal_raw)) / Decimal(1_000_000)

    if dry_run:
        return {
            "status": "ok",
            "dry_run": True,
            "from": from_wallet,
            "to": to_wallet,
            "condition_id": condition_id,
            "index_set": int(index_set),
            "collateral": collateral_used,
            "token_id": str(position_id),
            "raw_balance": int(bal_raw),
            "shares": float(shares),
            "message": "dry_run",
        }

    try:
        nonce = await w3.eth.get_transaction_count(from_wallet, "pending")
        tx = await conditional.functions.safeTransferFrom(
            from_wallet,
            w3.to_checksum_address(to_wallet),
            int(position_id),
            int(bal_raw),
            b"",
        ).build_transaction(
            {
                "from": from_wallet,
                "nonce": nonce,
                "chainId": int(getattr(Config, "CHAIN_ID", 137) or 137),
                "gas": int(max_gas),
                "gasPrice": await w3.eth.gas_price,
            }
        )
        signed = w3.eth.account.sign_transaction(tx, private_key=Config.POLYMARKET_PRIVATE_KEY)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        tx_hash = await w3.eth.send_raw_transaction(raw)
        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        ok = int(getattr(receipt, "status", 0) or 0) == 1
        return {
            "status": "ok",
            "dry_run": False,
            "from": from_wallet,
            "to": to_wallet,
            "condition_id": condition_id,
            "index_set": int(index_set),
            "collateral": collateral_used,
            "token_id": str(position_id),
            "raw_balance": int(bal_raw),
            "shares": float(shares),
            "tx_hash": tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash),
            "tx_status": "sent" if ok else "failed",
        }
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "condition_id": condition_id,
                "index_set": int(index_set),
                "token_id": str(position_id),
            },
            status_code=502,
        )


@app.post("/api/fetch-orphaned-tokens")
async def fetch_orphaned_tokens(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny

    from config import Config
    from eth_account import Account

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not Config.POLYGON_RPC_URL:
        return JSONResponse({"status": "error", "message": "POLYGON_RPC_URL not set"}, status_code=503)
    if not Config.POLYMARKET_PRIVATE_KEY:
        return JSONResponse({"status": "error", "message": "POLYMARKET_PRIVATE_KEY not set"}, status_code=503)

    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    from_wallet = account.address

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        return JSONResponse({"status": "error", "message": "Could not connect to Polygon RPC"}, status_code=503)

    CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CONDITIONAL_ABI = [
        {
            "inputs": [
                {"internalType": "address", "name": "account", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
            ],
            "name": "balanceOf",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]

    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=CONDITIONAL_ABI)

    def _hex_topic_addr(addr: str) -> str:
        a = addr.lower().replace("0x", "")
        return "0x" + ("0" * 24) + a

    lookback_blocks = int((body or {}).get("lookback_blocks", 2_000_000) or 2_000_000)

    try:
        latest = await w3.eth.block_number
    except Exception:
        latest = await w3.eth.get_block_number()

    start = max(0, int(latest) - int(lookback_blocks))
    end = int(latest)

    cs = int(chunk_size or 0)
    if cs <= 0:
        cs = 200_000

    diagnostics: dict = {
        "source": "onchain_logs",
        "from_block": int(start),
        "to_block": int(end),
        "lookback_blocks": int(lookback_blocks),
        "chunk_size": int(cs),
        "log_queries": 0,
        "logs_returned": 0,
        "getlogs_errors": [],
    }

    wallet_topic = _hex_topic_addr(from_wallet)
    sig_single = "0x" + w3.keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex().replace("0x", "")
    sig_batch = "0x" + w3.keccak(text="TransferBatch(address,address,address,uint256[],uint256[])").hex().replace("0x", "")

    async def fetch_logs_chunked(sig: str, topics: list[Optional[str]]):
        out_logs = []
        cur = int(start)
        while cur <= int(end):
            to_b = min(int(end), cur + cs - 1)
            diagnostics["log_queries"] += 1
            try:
                chunk = await w3.eth.get_logs(
                    {
                        "fromBlock": cur,
                        "toBlock": to_b,
                        "address": w3.to_checksum_address(CONDITIONAL_TOKENS),
                        "topics": topics,
                    }
                )
                if chunk:
                    out_logs.extend(chunk)
            except Exception as e:
                diagnostics["getlogs_errors"].append(
                    {
                        "fromBlock": int(cur),
                        "toBlock": int(to_b),
                        "topic0": str(sig),
                        "topics": [str(t) if t is not None else None for t in topics],
                        "error": str(e),
                    }
                )
            cur = to_b + 1
        return out_logs

    logs = []
    logs += await fetch_logs_chunked(sig_single, [sig_single, None, None, wallet_topic])
    logs += await fetch_logs_chunked(sig_single, [sig_single, None, wallet_topic, None])
    logs += await fetch_logs_chunked(sig_batch, [sig_batch, None, None, wallet_topic])
    logs += await fetch_logs_chunked(sig_batch, [sig_batch, None, wallet_topic, None])

    diagnostics["logs_returned"] = int(len(logs))

    token_ids: set[int] = set()
    for lg in logs:
        try:
            t0 = (lg.get("topics") or [])[0]
            if isinstance(t0, (bytes, bytearray)):
                t0 = "0x" + bytes(t0).hex()

            data_hex = lg.get("data") or "0x"
            if isinstance(data_hex, (bytes, bytearray)):
                data = bytes(data_hex)
            else:
                data = bytes.fromhex(str(data_hex).replace("0x", ""))

            if str(t0).lower() == str(sig_single).lower():
                if len(data) >= 64:
                    token_ids.add(int.from_bytes(data[0:32], "big"))
            elif str(t0).lower() == str(sig_batch).lower():
                try:
                    from eth_abi import decode

                    ids, _values = decode(["uint256[]", "uint256[]"], data)
                    for tid in ids:
                        token_ids.add(int(tid))
                except Exception:
                    continue
        except Exception:
            continue

    orphaned_tokens: list[str] = []
    for tid in token_ids:
        try:
            bal_raw = await conditional.functions.balanceOf(from_wallet, int(tid)).call()
            if bal_raw and int(bal_raw) > 0:
                orphaned_tokens.append(str(int(tid)))
        except Exception:
            continue

    orphaned_tokens = list(set(orphaned_tokens))

    diagnostics["candidate_token_ids"] = int(len(token_ids))
    diagnostics["held_token_ids"] = int(len(orphaned_tokens))

    return {
        "status": "ok",
        "orphaned_tokens": orphaned_tokens,
        "count": len(orphaned_tokens),
        "source": "onchain_logs",
        "lookback_blocks": int(lookback_blocks),
        "log_chunk": int(cs),
        "from_block": int(start),
        "to_block": int(end),
        "diagnostics": diagnostics,
    }


@app.post("/api/manual/exit-limit")
async def manual_exit_limit(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    try:
        data = await request.json()
        price = float((data or {}).get("price"))
        order_type = str((data or {}).get("order_type") or "GTC").upper()
        if order_type not in ("GTC", "FOK", "IOC"):
            order_type = "GTC"
        result = await engine.enqueue_manual_exit_limit(price=price, order_type=order_type)
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/api/manual/exit-replace")
async def manual_exit_replace(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    try:
        data = await request.json()
        price = float((data or {}).get("price"))
        result = await engine.enqueue_manual_exit_replace(price=price)
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/api/manual/exit-cancel")
async def manual_exit_cancel(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    try:
        result = await engine.enqueue_manual_exit_cancel()
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/api/force-clear-position")
async def force_clear_position(request: Request):
    """Emergency: clear held_position state when shares are already sold but state is stuck."""
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine or not engine.state:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    from state import HeldPosition
    old = engine.state.held_position
    info = f"Cleared: side={old.side} size={old.size} pending={old.is_pending} exit_reason={old.exit_reason}"
    engine.state.held_position = HeldPosition()
    engine.state.pos_current_price = None
    engine.state.pos_unrealized_pnl_pct = None
    engine.state.pos_unrealized_pnl_usd = None
    await engine.state_mgr.save(engine.state)
    log.warning(f"FORCE_CLEAR_POSITION via API: {info}")
    return {"status": "ok", "message": info}


@app.post("/api/fix-last-trade")
async def fix_last_trade(request: Request):
    """Fix the most recent trade's exit price/pnl when it was wrongly marked (e.g., -100% LOSS on a manual exit)."""
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine or not engine.state:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    body = await request.json()
    exit_price = float(body.get("exit_price", 0))
    if exit_price <= 0 or exit_price > 1.0:
        return JSONResponse({"status": "error", "message": "exit_price must be between 0.01 and 1.00"}, status_code=400)
    # Find the most recent trade (or match by side if provided)
    side = body.get("side")
    target = None
    for tr in reversed(engine.state.trade_history):
        if side and tr.side != side:
            continue
        target = tr
        break
    if not target:
        return JSONResponse({"status": "error", "message": "No matching trade found"}, status_code=404)
    old_exit = target.exit_price
    old_pnl = target.pnl
    old_outcome = target.outcome
    entry_px = target.entry_price or 0.5
    target.exit_price = exit_price
    target.pnl = (exit_price - entry_px) / entry_px if entry_px > 0 else 0.0
    target.outcome = "WIN" if target.pnl >= 0 else "LOSS"
    await engine.state_mgr.save(engine.state)
    log.warning(f"FIX_LAST_TRADE via API: {target.side} entry={entry_px:.3f} exit {old_exit}->{exit_price} pnl {old_pnl}->{target.pnl:.4f} outcome {old_outcome}->{target.outcome}")
    return {
        "status": "ok",
        "entry": entry_px,
        "exit_price": exit_price,
        "pnl_pct": round(target.pnl * 100, 2),
        "outcome": target.outcome,
    }


@app.post("/api/repair-trade")
async def repair_trade(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine or not engine.state or not engine.state_mgr:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        body = {}

    market_slug = str((body or {}).get("market_slug") or "").strip()
    dry_run = bool((body or {}).get("dry_run", False))
    if not market_slug:
        return JSONResponse({"status": "error", "message": "market_slug is required"}, status_code=400)

    from config import Config
    from eth_account import Account

    try:
        wallet = Account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Could not derive wallet from config: {e}"}, status_code=500)

    target = None
    for tr in reversed(engine.state.trade_history):
        if getattr(tr, "exit_reason", None) == "AUTO_SETTLE_LOSS" and _safe_float(getattr(tr, "pnl", 0.0)) > 0:
            target = tr
            break
    if target is None:
        return JSONResponse(
            {"status": "error", "message": "No positive-PnL AUTO_SETTLE_LOSS trade found to repair"},
            status_code=404,
        )

    try:
        repair_inputs = await _fetch_trade_repair_inputs(wallet, market_slug)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=502)

    buy_ts = int((repair_inputs.get("buy") or {}).get("timestamp") or 0)
    if getattr(target, "ts", None) and buy_ts and abs(int(target.ts) - buy_ts) > 3600:
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    f"Candidate trade timestamp {int(target.ts)} does not align with "
                    f"matched BUY timestamp {buy_ts} for {market_slug}"
                ),
            },
            status_code=409,
        )

    trade_before = _serialize_trade_record(target)
    state_before = _summarize_trade_history(engine.state.trade_history)
    repaired = _build_repaired_trade(target, repair_inputs)

    def _apply_trade_snapshot(trade, snapshot: dict):
        trade.size = snapshot["size"]
        trade.entry_price = snapshot["entry_price"]
        trade.exit_price = snapshot["exit_price"]
        trade.pnl = snapshot["pnl"]
        trade.outcome = snapshot["outcome"]
        trade.exit_reason = snapshot["exit_reason"]
        trade.partial_exits = list(snapshot["partial_exits"])

    _apply_trade_snapshot(target, repaired)
    state_after = _summarize_trade_history(engine.state.trade_history)
    trade_after = _serialize_trade_record(target)

    if dry_run:
        _apply_trade_snapshot(target, trade_before)
        return {
            "status": "ok",
            "dry_run": True,
            "matched_activity": {
                "buy": repair_inputs["buy"],
                "sells": repair_inputs["sells"],
                "winning_outcome": repair_inputs["winning_outcome"],
                "side_won": repaired["side_won"],
                "remaining_size": repaired["remaining_size"],
                "settle_price": repaired["settle_price"],
            },
            "trade_before": trade_before,
            "trade_after": trade_after,
            "state_counter_diff": {
                "before": state_before,
                "after": state_after,
                "delta": {
                    k: round(state_after[k] - state_before[k], 6)
                    for k in state_before.keys()
                },
            },
            "closed_trade_updates": {
                "candidate_ids": [
                    r["id"] for r in await engine.state_mgr.fetch_closed_trades(market_slug=market_slug)
                ],
            },
            "duplicates_removed": 0,
        }

    existing_rows = await engine.state_mgr.fetch_closed_trades(market_slug=market_slug)
    engine.rebuild_state_totals_from_trade_history()
    canonical_row = engine._closed_trade_row_from_record(
        target,
        market_slug,
        timestamp=(existing_rows[0]["timestamp"] if existing_rows else int(time.time())),
    )
    if existing_rows:
        latest = existing_rows[0]
        for key in ("slippage", "regime", "features", "kelly_fraction"):
            if canonical_row.get(key) in (None, "") and latest.get(key) not in (None, ""):
                canonical_row[key] = latest.get(key)

    sync_result = await engine.state_mgr.resync_closed_trades_from_canonical(
        [canonical_row],
        market_slug=market_slug,
        delete_extra=True,
    )
    await engine.state_mgr.calculate_performance_metrics()
    await engine.state_mgr.save(engine.state)

    return {
        "status": "ok",
        "dry_run": False,
        "matched_activity": {
            "buy": repair_inputs["buy"],
            "sells": repair_inputs["sells"],
            "winning_outcome": repair_inputs["winning_outcome"],
            "side_won": repaired["side_won"],
            "remaining_size": repaired["remaining_size"],
            "settle_price": repaired["settle_price"],
        },
        "trade_before": trade_before,
        "trade_after": trade_after,
        "state_counter_diff": {
            "before": state_before,
            "after": state_after,
            "delta": {
                k: round(state_after[k] - state_before[k], 6)
                for k in state_before.keys()
            },
        },
        "closed_trade_updates": sync_result,
        "duplicates_removed": int(sync_result.get("deleted", 0)),
    }


@app.post("/api/manual/exit-now")
async def manual_exit_now(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    try:
        result = await engine.enqueue_manual_exit_now()
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/api/manual/entry")
async def manual_entry(request: Request):
    deny = _require_admin(request)
    if deny:
        return deny
    if not engine:
        return JSONResponse({"status": "error", "message": "Engine not ready"}, status_code=503)
    try:
        data = await request.json()
        side = str((data or {}).get("side", "")).upper()
        if side not in ("YES", "NO"):
            return JSONResponse({"status": "error", "message": "Side must be YES or NO"}, status_code=400)
        price = float((data or {}).get("price", 0))
        size_usd = float((data or {}).get("size_usd", 0))
        order_type = str((data or {}).get("order_type", "GTC")).upper()
        if order_type not in ("GTC", "FOK"):
            order_type = "GTC"
        if price <= 0 or price >= 1:
            return JSONResponse({"status": "error", "message": "Price must be between 0.01 and 0.99"}, status_code=400)
        if size_usd < 1:
            return JSONResponse({"status": "error", "message": "Size must be >= $1"}, status_code=400)
        result = await engine.enqueue_manual_entry(side=side, price=price, size_usd=size_usd, order_type=order_type)
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


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
    try:
        from main import BUILD_VERSION
    except Exception:
        BUILD_VERSION = "unknown"
    # Always try to load heartbeat file first — available even during startup
    hb_path = "/data/heartbeat.json" if os.path.isdir("/data") else "heartbeat.json"
    hb = {}
    if os.path.exists(hb_path):
        try:
            with open(hb_path, 'r') as f:
                hb = json.load(f)
        except: pass

    # If engine isn't ready yet, serve heartbeat data — but flag it as stale if
    # the heartbeat file hasn't been updated in > 30s (engine has crashed).
    if not engine or not engine.state or not getattr(engine, "_running", False):
        hb_age = time.time() - os.path.getmtime(hb_path) if os.path.exists(hb_path) else 9999
        if hb_age > 30 and hb:
            hb["status"] = "crashed"
            hb["engine_stale"] = True
            hb["hb_age_sec"] = int(hb_age)
            return JSONResponse(content=hb, status_code=503)
        hb.setdefault("status", "initializing")
        hb.setdefault("wallet_usdc", hb.get("balance", 0.0))
        # Alias performance → performance_metrics so the JS finds it
        if "performance" in hb and "performance_metrics" not in hb:
            hb["performance_metrics"] = hb["performance"]
        return hb

    state = engine.state

    if not hb and engine.pm:
        try:
            hb["wallet_usdc"] = await engine.pm.get_wallet_usdc_balance() or 0.0
        except: pass

    balance = hb.get("wallet_usdc") or hb.get("balance", 0.0)
    open_pos = 1 if state.held_position.side else 0
    exposure = state.held_position.size_usd if state.held_position.side else 0.0

    if state.held_position.side:
        hb["position"] = engine.state.held_position.side
        hb["entry_price"] = engine.state.held_position.avg_entry_price or engine.state.held_position.entry_price
        hb["position_size"] = engine.state.held_position.size if engine.state.held_position.size else 0.0
        hb["tx_hash"] = engine.state.trade_history[-1].tx_hash if engine.state.trade_history else None

    # Base payload on heartbeat, then overlay live state
    metrics = hb.copy()
    metrics.update({
        "balance": hb.get("balance", 0.0),
        "status": "Running" if getattr(engine, "_running", False) else "Stopped",
        "uptime_sec": int(time.time() - start_time),
        "version": BUILD_VERSION,
        "wallet_usdc": balance,
        "open_positions": open_pos,
        "exposure_usd": exposure,
        "strike_source": getattr(state, "strike_source", "none"),
        "trading_halted": getattr(state, "trading_halted", False),
        "total_pnl_usd": state.total_pnl_usd,
        "total_trades": state.total_trades,
        "total_wins": state.total_wins,
        "total_losses": state.total_losses,
        "last_market_slug": getattr(state, "last_market_slug", ""),
        "last_market_expiry": getattr(state, "last_market_expiry", None),
        "latencies": getattr(state, "latencies", {}) or hb.get("latencies") or {},
        "performance_metrics": getattr(state, "performance_metrics", None) or hb.get("performance") or {},
        "last_tuned_time": getattr(state, "last_tuned_time", 0),
        "pos_current_price": getattr(state, "pos_current_price", None),
        "pos_unrealized_pnl_pct": getattr(state, "pos_unrealized_pnl_pct", None),
        "pos_unrealized_pnl_usd": getattr(state, "pos_unrealized_pnl_usd", None),
        "open_positions_api": getattr(state, "open_positions_api", []),
        "recent_trades": [
            {
                "ts": t.ts,
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "outcome": t.outcome,
                "slippage": getattr(t, "slippage", None),
                "size": getattr(t, "size", None),
                "tx_hash": getattr(t, "tx_hash", None),
                "exit_reason": getattr(t, "exit_reason", None),
                "partial_exits": getattr(t, "partial_exits", []),
            }
            for t in state.trade_history[-10:]
        ]
    })

    signal = metrics.get("signal") or {}
    skip_gates = signal.get("skip_gates") or []
    metrics["gate_primary"] = skip_gates[0] if skip_gates else "CLEAR"
    metrics["gate_all"] = skip_gates
    metrics["sigma_b"] = signal.get("sigma_b")
    metrics["bvol_multiplier"] = signal.get("bvol_multiplier")

    # Back-compat alias for older frontend references
    if "performance" not in metrics:
        metrics["performance"] = metrics.get("performance_metrics") or {}

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
    data["type"] = "metrics"
    await ws_manager.broadcast(data)


@app.get("/api/events")
async def get_events(limit: int = 50):
    """Return recent events for clients that missed WS broadcasts."""
    return list(_event_buffer)[:limit]


@app.get("/api/push/vapid-public-key")
async def push_vapid_public_key():
    return {"key": _VAPID_PUBLIC}

@app.post("/api/push/subscribe")
async def push_subscribe(request: Request):
    sub = await request.json()
    _push_subscriptions[sub["endpoint"]] = sub
    return {"ok": True, "count": len(_push_subscriptions)}

@app.delete("/api/push/subscribe")
async def push_unsubscribe(request: Request):
    sub = await request.json()
    _push_subscriptions.pop(sub.get("endpoint", ""), None)
    return {"ok": True}


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


# ── Audit 3 P5: Optimizer detail for dashboard self-learning widget ──────────
@app.get("/api/optimizer-detail")
async def optimizer_detail():
    if not engine:
        return JSONResponse({"status": "loading"}, status_code=503)
    try:
        detail = engine.optimizer.get_optimizer_detail()
        # Add optimizer status label
        closed_count = len([t for t in engine.state.trade_history if t.outcome in ("WIN", "LOSS")])
        if closed_count < 1:
            detail["status"] = "AWAITING"
        elif closed_count < 10:
            detail["status"] = "COLLECTING"
        else:
            detail["status"] = "LEARNING"
        detail["closed_trades"] = closed_count
        return detail
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


@app.get("/api/exit-stats")
async def exit_stats():
    """Phase A: summarize exit_outcomes.jsonl by reason — early exit regret rate."""
    import json, os
    path = "/data/exit_outcomes.jsonl" if os.path.exists("/data") else "exit_outcomes.jsonl"
    if not os.path.exists(path):
        return {"message": "No exit records yet", "records": 0}
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    by_reason: dict = {}
    for r in records:
        reason = r.get("exit_reason", "UNKNOWN")
        if reason not in by_reason:
            by_reason[reason] = {"count": 0, "settled": 0, "regret": 0, "unrealized_pct_sum": 0.0}
        bucket = by_reason[reason]
        bucket["count"] += 1
        bucket["unrealized_pct_sum"] += r.get("unrealized_pct", 0.0)
        if r.get("settlement_itm") is not None:
            bucket["settled"] += 1
            # "regret" = we exited but it settled ITM (we could have held for profit)
            if r["settlement_itm"] and r.get("unrealized_pct", 0) < 0:
                bucket["regret"] += 1

    summary = {}
    for reason, b in by_reason.items():
        summary[reason] = {
            "count": b["count"],
            "avg_unrealized_pct": round(b["unrealized_pct_sum"] / b["count"] * 100, 2) if b["count"] else 0,
            "settled_count": b["settled"],
            "regret_count": b["regret"],
            "regret_rate": round(b["regret"] / b["settled"], 3) if b["settled"] else None,
        }
    return {"records": len(records), "by_reason": summary}


# ═══════════════════════════ Backtest UI + API ═══════════════════════════════
#
# Unified backtest replay harness for the Backtest page. Runs in a background
# thread so the dashboard event loop stays responsive; results persist to
# data/backtests/{run_id}.json for the "recent runs" dropdown.

import uuid as _uuid
import threading as _threading

_backtest_runs: dict = {}   # run_id -> {"status": "..", "progress": 0-1, "message": "..", "error": None}
_backtest_lock = _threading.Lock()


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    try:
        return templates.TemplateResponse("backtest.html", {"request": request})
    except Exception as e:
        log.exception("Failed to render backtest.html")
        return HTMLResponse(f"<h1>Backtest template error</h1><pre>{e}</pre>", status_code=500)


def _run_backtest_bg(run_id: str, payload: dict):
    """Background worker. Guarded by try/except; always writes final status."""
    try:
        from backtest_engine import BacktestParams, run_backtest, save_run

        def _parse_date(s):
            if not s:
                return None
            try:
                from datetime import datetime
                return int(datetime.strptime(s, "%Y-%m-%d").timestamp())
            except Exception:
                return None

        params = BacktestParams(
            fill_model=payload.get("fill_model", "synthetic"),
            source=payload.get("source", "trade_features"),
            start_ts=_parse_date(payload.get("start_date")),
            end_ts=_parse_date(payload.get("end_date")),
            score_offset=float(payload.get("score_offset", 0) or 0),
            edge_offset=float(payload.get("edge_offset", 0) or 0),
            flags=payload.get("flags", {}) or {},
        )

        def _cb(frac, msg):
            with _backtest_lock:
                r = _backtest_runs.get(run_id)
                if r is not None:
                    r["progress"] = float(frac)
                    r["message"]  = msg

        result = run_backtest(params, progress_cb=_cb)
        save_run(run_id, result)
        with _backtest_lock:
            _backtest_runs[run_id] = {
                "status": "done", "progress": 1.0,
                "message": f"{result['metrics'].get('n_trades', 0)} trades",
                "error": None,
            }
    except Exception as e:
        log.exception("backtest run failed")
        with _backtest_lock:
            _backtest_runs[run_id] = {
                "status": "error", "progress": 1.0,
                "message": str(e), "error": str(e),
            }


@app.post("/api/backtest/run")
async def backtest_run(request: Request):
    fail = _require_admin(request)
    if fail:
        return fail
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    run_id = _uuid.uuid4().hex[:12]
    with _backtest_lock:
        _backtest_runs[run_id] = {
            "status": "running", "progress": 0.0,
            "message": "queued", "error": None,
        }
    t = _threading.Thread(target=_run_backtest_bg, args=(run_id, payload), daemon=True)
    t.start()
    return {"run_id": run_id}


@app.get("/api/backtest/status/{run_id}")
async def backtest_status(run_id: str, request: Request):
    fail = _require_admin(request)
    if fail:
        return fail
    with _backtest_lock:
        r = _backtest_runs.get(run_id)
    if r is None:
        # Completed run that persisted but evicted from in-memory map
        from backtest_engine import load_run
        if load_run(run_id) is not None:
            return {"status": "done", "progress": 1.0, "message": "persisted"}
        return JSONResponse({"status": "error", "message": "unknown run_id"}, status_code=404)
    return r


@app.get("/api/backtest/result/{run_id}")
async def backtest_result(run_id: str, request: Request):
    fail = _require_admin(request)
    if fail:
        return fail
    from backtest_engine import load_run
    result = load_run(run_id)
    if result is None:
        return JSONResponse({"status": "error", "message": "not ready"}, status_code=404)
    return result


@app.get("/api/backtest/list")
async def backtest_list(request: Request):
    fail = _require_admin(request)
    if fail:
        return fail
    from backtest_engine import list_runs
    return {"runs": list_runs(30)}


async def run_dashboard(engine_instance, port=8000):
    global engine
    engine = engine_instance
    import uvicorn
    # log_config=None lets uvicorn reset the root logger, swallowing engine/signals logs.
    # Instead pass a minimal config that keeps uvicorn's access log but leaves all other
    # loggers (engine, signals, sizing, etc.) owned by our setup_logging() basicConfig.
    _log_config = {
        "version": 1,
        "disable_existing_loggers": False,  # critical — preserves engine/signals loggers
        "formatters": {
            "default": {
                "fmt": "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn.error":  {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_config=_log_config)
    server = uvicorn.Server(config)
    
    # We must run `serve` as a task, but await it so it occupies this background task
    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
