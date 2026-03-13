import asyncio
from web3 import AsyncWeb3
from config import Config

# Addresses (Polygon mainnet)
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ERC-1155 safeTransferFrom signature
# safeTransferFrom(address from, address to, uint256 id, uint256 amount, bytes data)
ERC1155_SAFE_TRANSFER_FROM = "0xf242432a"  # safeTransferFrom(address,address,uint256,uint256,bytes)

async def main():
    if not Config.POLYGON_RPC_URL:
        print("Error: POLYGON_RPC_URL not set in config.py")
        return
    if not Config.POLYMARKET_PRIVATE_KEY:
        print("Error: POLYMARKET_PRIVATE_KEY not set in config.py")
        return

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        print("Error: Could not connect to Polygon RPC.")
        return

    # EOA that holds the orphan YES token
    from_account = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    # Proxy wallet used by Polymarket (from UI)
    to_account = w3.to_checksum_address("0xddb76ec1164a72d01211524a0a056bc9c1d8574c")

    # YES token ID for this market
    yes_token_id = 104396440751277512071663049096787928567222444925008417395090357720411752196622
    amount = 979080  # your YES balance

    print(f"Transferring {amount} YES tokens from {from_account.address} to {to_account}")

    # Build the safeTransferFrom call using the contract ABI instead of manual encoding
    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=[
        {
            "inputs": [
                {"internalType": "address", "name": "from", "type": "address"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "id", "type": "uint256"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"},
                {"internalType": "bytes", "name": "data", "type": "bytes"}
            ],
            "name": "safeTransferFrom",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ])

    try:
        tx = await conditional.functions.safeTransferFrom(
            from_account.address,
            to_account,
            yes_token_id,
            amount,
            b''
        ).build_transaction({
            'from': from_account.address,
            'nonce': await w3.eth.get_transaction_count(from_account.address),
            'gas': 200000,
            'gasPrice': await w3.eth.gas_price,
        })
        signed = from_account.sign_transaction(tx)
        tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"Tx sent: {tx_hash.hex()}")

        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status == 1:
            print("✅ Transfer succeeded! Now check Polymarket UI with the proxy wallet.")
        else:
            print(f"❌ Transfer failed (status {receipt.status})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
