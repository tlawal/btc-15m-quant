import asyncio
import os
import sqlite3
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Database path handling (same as state.py fallback)
DB_PATH = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./state.db")
if not DB_PATH.startswith("sqlite+aiosqlite:///"):
    DB_PATH = "sqlite+aiosqlite:///./state.db"
    
if os.path.isdir("/data"): # Railway volume
    DB_PATH = "sqlite+aiosqlite:////data/state.db"

async def fix_last_trade():
    print(f"Connecting to database: {DB_PATH}")
    engine = create_async_engine(DB_PATH)
    
    async with engine.begin() as conn:
        # Find the most recent trade that had a -100% loss (often recorded as exit_price=0 or DUST_WRITEOFF)
        # Specifically looking for the trade from the logs you provided
        query = text("""
            SELECT id, timestamp, market_slug, size, entry_price, exit_price, pnl_usd, outcome_win 
            FROM closed_trades 
            ORDER BY timestamp DESC LIMIT 5
        """)
        
        result = await conn.execute(query)
        rows = result.fetchall()
        
        print("\n--- Recent Trades ---")
        target_id = None
        for r in rows:
            print(f"ID: {r.id} | TS: {r.timestamp} | Size: {r.size} | Entry: {r.entry_price} | Exit: {r.exit_price} | PNL: ${r.pnl_usd:.2f} | Win: {r.outcome_win}")
            # Target the specific trade (around 13:07/13:08 UTC, large negative PNL, entry $0.85)
            if r.entry_price == 0.85 and r.pnl_usd < -1.0:
                target_id = r.id
                
        if not target_id:
            print("\nCould not automatically identify the trade to fix (looking for Entry: 0.85 and large negative PNL).")
            print("Please run this script again and manually input the ID you want to fix.")
            return

        print(f"\n=> Found target trade ID to fix: {target_id}")
        
        # Calculate correct PNL based on 7.96 shares sold at $0.87
        actual_profit = (0.87 - 0.85) * 7.96
        
        # Update closed_trades table
        update_query = text("""
            UPDATE closed_trades 
            SET exit_price = 0.87, pnl_usd = :profit, outcome_win = 1, exit_reason = 'FIXED_RECONCILE' 
            WHERE id = :id
        """)
        
        await conn.execute(update_query, {"profit": actual_profit, "id": target_id})
        print(f"Successfully updated trade ID {target_id} to Win with profit +${actual_profit:.2f}")
        
        # Recalculate total_pnl_usd
        sum_query = text("SELECT SUM(pnl_usd) FROM closed_trades")
        total_pnl_result = await conn.execute(sum_query)
        new_total_pnl = total_pnl_result.scalar() or 0.0
        
        print(f"New Database Total PNL: ${new_total_pnl:.2f}")
        
        # We also need to fix the engine state's embedded `total_pnl_usd` inside the `kv` table
        import json
        kv_query = text("SELECT value FROM kv WHERE key = 'total_pnl_usd'")
        kv_result = await conn.execute(kv_query)
        kv_row = kv_result.fetchone()
        
        if kv_row:
            await conn.execute(
                text("UPDATE kv SET value = :val WHERE key = 'total_pnl_usd'"),
                {"val": str(round(new_total_pnl, 4))}
            )
            print("Successfully synced `kv` total_pnl_usd state marker.")
            
        print("\nFix applied successfully! Please restart your bot for the changes to take effect in memory/dashboard.")

if __name__ == "__main__":
    asyncio.run(fix_last_trade())
