import asyncio
from web3 import AsyncWeb3
from config import Config

async def get_condition_id_from_tx(tx_hash: str):
    """Get condition ID from a Polymarket transaction hash."""
    if not Config.POLYGON_RPC_URL:
        print("Error: POLYGON_RPC_URL is not set.")
        return

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        print("Error: Could not connect to Polygon RPC.")
        return

    try:
        # Get transaction receipt
        receipt = await w3.eth.get_transaction_receipt(tx_hash)
        if not receipt:
            print("Transaction not found")
            return

        # Look for condition ID in logs (Polymarket transactions log condition ID)
        for log in receipt.logs:
            if len(log.topics) > 0:
                # Condition ID is typically logged in Polymarket events
                # Look for bytes32 data
                if len(log.data) >= 64:  # bytes32 is 32 bytes = 64 hex chars
                    # First 32 bytes might be condition ID
                    condition_id = '0x' + log.data[2:66]  # skip 0x, take next 64 chars
                    print(f"Potential condition ID from log: {condition_id}")
                    return condition_id

        print("Condition ID not found in transaction logs")

    except Exception as e:
        print(f"Error fetching transaction: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python get_condition_from_tx.py <tx_hash>")
        sys.exit(1)

    tx_hash = sys.argv[1]
    asyncio.run(get_condition_id_from_tx(tx_hash))
