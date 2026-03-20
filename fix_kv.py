import asyncio
import os
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DB_PATH = "sqlite+aiosqlite:////data/state.db" if os.path.exists("/data/state.db") else "sqlite+aiosqlite:///./state.db"

async def fix():
    engine = create_async_engine(DB_PATH)
    async with engine.begin() as conn:
        res = await conn.execute(text("SELECT value FROM kv WHERE key = 'trade_history'"))
        row = res.fetchone()
        if not row: return
        history = json.loads(row[0])
        found = False
        for trade in history:
            if trade.get('entry_price') == 0.85 and (trade.get('pnl') or 0) < -0.9:
                trade['exit_price'] = 0.87
                trade['outcome'] = 'WIN'
                trade['pnl'] = (0.87 - 0.85) / 0.85
                found = True
                print("Patched trade_history list!")
        if found:
            await conn.execute(
                text("UPDATE kv SET value = :val WHERE key = 'trade_history'"),
                {"val": json.dumps(history)}
            )
            print("Successfully updated KV memory JSON.")

asyncio.run(fix())
