import os
import asyncio
from web3 import AsyncWeb3
from config import Config

# CTF Exchange on Polygon (v2)
CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"

# ABI for redeemPositions
CTF_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

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

    account = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_EXCHANGE), abi=CTF_ABI)

    # Your resolved market
    condition_id = "0xf89e490675e9aa6bf294f26d1bbfb76bcaef4afaf1a3222f216ecc828cdd2247"
    index_sets = [1, 2]  # binary markets: YES=1, NO=2

    print(f"Redeeming YES shares for condition {condition_id} from EOA {account.address}")

    try:
        tx = await ctf.functions.redeemPositions(
            w3.to_bytes(hexstr=condition_id),
            index_sets
        ).build_transaction({
            'from': account.address,
            'nonce': await w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': await w3.eth.gas_price
        })

        signed = account.sign_transaction(tx)
        tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"Tx sent: {tx_hash.hex()}")

        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status == 1:
            print("✅ Successfully redeemed YES shares!")
        else:
            print(f"❌ Tx failed (status {receipt.status})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
