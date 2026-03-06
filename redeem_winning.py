import os
import sys
import json
import asyncio
from typing import Optional

from web3 import AsyncWeb3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from config import Config

# Polymarket constants
POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# CTF Exchange on Polygon (v2)
CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"

# ABI for redeemPositions in CTF Exchange
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

async def redeem_winning_positions():
    """Find winning positions on Polymarket and redeem them directly via the CTF contract."""
    print("Initializing Polymarket Client...")
    
    secret = Config.POLYMARKET_API_SECRET or ""
    if secret and len(secret) % 4 != 0:
        secret += "=" * (4 - (len(secret) % 4))
        
    client = ClobClient(
        host=POLYMARKET_HOST,
        key=Config.POLYMARKET_PRIVATE_KEY,
        chain_id=Config.CHAIN_ID,
        creds=ApiCreds(
            api_key=Config.POLYMARKET_API_KEY,
            api_secret=secret,
            api_passphrase=Config.POLYMARKET_API_PASSPHRASE,
        ),
    )
    
    wallet_address = client.get_address()
    print(f"Trading wallet: {wallet_address}")
    
    # 1. Get raw positions via CLOB API to find redeemable ones
    print("\nFetching positions from CLOB API...")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        url = f"{POLYMARKET_HOST}/positions"
        headers = client.headers.get_headers() if hasattr(client, 'headers') else {}
        try:
            async with session.get(url, headers=headers) as r:
                if r.status != 200:
                    print(f"Error fetching from CLOB: {r.status} {await r.text()}")
                    return
                positions = await r.json()
        except Exception as e:
            print(f"CLOB API exception: {e}")
            return
            
    print(f"Found {len(positions)} total positions.")
    
    # Needs to match CLOB API format which uses slightly different keys than Gamma
    redeemable = []
    for p in positions:
        # Pnl checks if we have a resolved winning side, or size > 0 and closed but not redeemed
        # Actually, for the standalone script, let's just use the conditionId from the screenshot
        if p.get("conditionId") == "0xeb06a9643485cf2ef1f6c4c0bda5d00a12004d9b43c68383a17e0b57e7bbd76f" or float(p.get("size", 0)) > 1.0:
            if p.get("closed"):
                redeemable.append(p)

    
    if not redeemable:
        print("No redeemable winning positions found.")
        return
        
    print(f"Found {len(redeemable)} redeemable position(s) worth ~${sum(float(p.get('size',0)) for p in redeemable):.2f}")
    
    # 2. Connect to Polygon
    if not Config.POLYGON_RPC_URL:
        print("Error: POLYGON_RPC_URL is not set.")
        return
        
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        print("Error: Could not connect to Polygon RPC.")
        return
        
    account = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    ctf_contract = w3.eth.contract(address=w3.to_checksum_address(CTF_EXCHANGE), abi=CTF_ABI)
    
    print("\nAttempting redemptions...")
    
    for p in redeemable:
        condition_id = p.get('conditionId')
        market_slug = p.get('marketSlug')
        size = float(p.get('size', 0))
        
        print(f"\nRedeeming {size:.2f} shares in {market_slug} (Condition: {condition_id})")
        
        # Binary markets always use indexSets [1, 2]
        # indexSets is a bitmask: 1 = outcome 0 (YES), 2 = outcome 1 (NO)
        index_sets = [1, 2]
        
        try:
            # Prepare transaction
            tx_data = ctf_contract.functions.redeemPositions(
                w3.to_bytes(hexstr=condition_id),
                index_sets
            ).build_transaction({
                'from': account.address,
                'nonce': await w3.eth.get_transaction_count(account.address),
                'gas': 200000,
                'gasPrice': await w3.eth.gas_price
            })
            
            # Sign & Send
            signed_tx = account.sign_transaction(tx_data)
            tx_hash = await w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            print(f"Transaction sent! Hash: {tx_hash.hex()}")
            print(f"Waiting for confirmation...")
            
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                print(f"✅ Successfully redeemed {size:.2f} USDC!")
            else:
                print(f"❌ Transaction failed. Status: {receipt.status}")
                
        except Exception as e:
            print(f"Error executing transaction: {e}")

if __name__ == "__main__":
    asyncio.run(redeem_winning_positions())
