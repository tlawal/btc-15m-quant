import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

async def test_taapi():
    key = os.getenv("TAAPI_KEY")
    if not key:
        print("No TAAPI_KEY found")
        return

    payloads = [
        # Variant 1: interval in construct
        {
            "secret": key,
            "construct": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "interval": "5m",
                "indicators": [
                    {"id": "rsi1", "indicator": "rsi"}
                ]
            }
        },
        # Variant 2: interval in indicator only
        {
            "secret": key,
            "construct": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "indicators": [
                    {"id": "rsi2", "indicator": "rsi", "interval": "5m"}
                ]
            }
        },
        # Variant 3: Root level (unlikely but worth a try)
        {
            "secret": key,
            "interval": "5m",
            "construct": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "indicators": [
                    {"id": "rsi3", "indicator": "rsi"}
                ]
            }
        }
    ]

    async with aiohttp.ClientSession() as session:
        for i, p in enumerate(payloads):
            print(f"Testing Variant {i+1}...")
            async with session.post("https://api.taapi.io/bulk", json=p) as r:
                body = await r.text()
                print(f"Status: {r.status}")
                print(f"Body: {body[:200]}")
                print("-" * 20)

if __name__ == "__main__":
    asyncio.run(test_taapi())
