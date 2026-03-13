import asyncio
from py_clob_client import ClobClient
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

    try:
        # Get the user’s address (this is the EOA)
        resp = await client.get_me()
        print("Me (EOA):", resp)
    except Exception as e:
        print(f"Failed to get me: {e}")

    # The proxy wallet is not directly exposed via the client, but you can find it in the UI or via a balance call
    # Alternatively, check your Polymarket profile page for “Wallet address”
    print("\nTo find your proxy wallet address:")
    print("1. Log in to Polymarket UI")
    print("2. Go to Profile > Wallet (or Settings)")
    print("3. The address shown there is the proxy wallet (different from your EOA)")

if __name__ == "__main__":
    asyncio.run(main())
