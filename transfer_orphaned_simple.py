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

# CTF Exchange on Polygon (v2) - ERC1155 contract
CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"

# ABI for CTF Exchange (ERC1155)
CTF_ABI = [
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
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "uint256", "name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256", "name": "indexSet", "type": "uint256"}
        ],
        "name": "getCollectionId",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    }
]

async def transfer_orphaned_position(condition_id: str, index_set: int, amount: float, proxy_wallet: str):
    """Transfer orphaned position tokens to the proxy wallet."""
    print(f"Transferring {amount} shares (conditionId={condition_id}, indexSet={index_set}) to {proxy_wallet}")

    # Connect to Polygon
    if not Config.POLYGON_RPC_URL:
        print("Error: POLYGON_RPC_URL is not set.")
        return

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        print("Error: Could not connect to Polygon RPC.")
        return

    account = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    from_address = account.address
    print(f"From address: {from_address}")

    ctf_contract = w3.eth.contract(address=w3.to_checksum_address(CTF_EXCHANGE), abi=CTF_ABI)

    # USDC collateral token (bridged USDC on Polygon)
    collateral_token = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e
    parent_collection_id = "0x0000000000000000000000000000000000000000000000000000000000000000"

    # Get position ID (collectionId)
    try:
        position_id_bytes = await ctf_contract.functions.getCollectionId(
            w3.to_checksum_address(collateral_token),
            parent_collection_id,
            w3.to_bytes(hexstr=condition_id),
            index_set
        ).call()
        position_id = int(position_id_bytes.hex(), 16)
        print(f"Position ID: {position_id}")
    except Exception as e:
        print(f"Error getting position ID: {e}")
        return

    # Check balance
    try:
        balance = await ctf_contract.functions.balanceOf(from_address, position_id).call()
        print(f"Current balance: {balance / 1e6:.2f} shares")  # Assuming 6 decimals
        if balance < amount * 1e6:
            print(f"Insufficient balance: have {balance / 1e6:.2f}, need {amount}")
            return
    except Exception as e:
        print(f"Error checking balance: {e}")
        return

    # Transfer
    amount_wei = int(amount * 1e6)  # 6 decimals
    to_address = w3.to_checksum_address(proxy_wallet)

    try:
        tx_data = ctf_contract.functions.safeTransferFrom(
            from_address,
            to_address,
            position_id,
            amount_wei,
            b""  # empty data
        ).build_transaction({
            'from': from_address,
            'nonce': await w3.eth.get_transaction_count(from_address),
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
            print(f"✅ Successfully transferred {amount} shares to {proxy_wallet}!")
        else:
            print(f"❌ Transaction failed. Status: {receipt.status}")

    except Exception as e:
        print(f"Error executing transaction: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python transfer_orphaned.py <condition_id> <index_set> <amount> <proxy_wallet>")
        print("Example: python transfer_orphaned.py 0x149a361bd4e2eebe4fdb2c7a33116debe6ff75c1fa4189255ea30bd4c1800ae7 2 1 0xc9068a1fb55d29e3c0dfbfcd537ee28983f6d62d981383c803988b7ce282eaa4")
        sys.exit(1)

    condition_id = sys.argv[1]
    index_set = int(sys.argv[2])
    amount = float(sys.argv[3])
    proxy_wallet = sys.argv[4]

    asyncio.run(transfer_orphaned_position(condition_id, index_set, amount, proxy_wallet))
