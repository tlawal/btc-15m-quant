import asyncio
from py_clob_client import ClobClient
from py_clob_client.constants import POLYGON
from config import Config

async def main():
    if not Config.POLYMARKET_PRIVATE_KEY:
        print("Error: POLYMARKET_PRIVATE_KEY not set in config.py")
        return

    client = ClobClient(
        host=Config.POLYMARKET_HOST,
        key=Config.POLYMARKET_PRIVATE_KEY,
        chain_id=Config.CHAIN_ID,
        signature_type=2,  # EIP712
        funder=Config.FUNDER_ADDRESS or None
    )

    # Your market token IDs (YES is first in clobTokenIds)
    yes_token_id = "104396440751277512071663049096787928567222444925008417395090357720411752196622"

    # Get current order book to pick a price
    try:
        book = await client.get_order_book(token_id=yes_token_id)
        if not book or not book.get("bids"):
            print("No bids available; cannot sell.")
            return
        best_bid = float(book["bids"][0][0])  # highest bid price
        print(f"Best bid price: {best_bid}")
    except Exception as e:
        print(f"Failed to fetch order book: {e}")
        return

    # Place a market sell (FOK) for the full balance
    size = 979080  # your YES balance
    try:
        resp = await client.market_sell(yes_token_id, size)
        print("Market sell placed:", resp)
    except Exception as e:
        print(f"Market sell failed: {e}")
        # If market sell fails (e.g., due to allowance), try limit sell at best bid
        try:
            resp = await client.limit_sell(yes_token_id, size, best_bid)
            print("Limit sell placed at best bid:", resp)
        except Exception as e2:
            print(f"Limit sell also failed: {e2}")

if __name__ == "__main__":
    asyncio.run(main())
