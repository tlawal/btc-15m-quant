import os
import sys
import json
import asyncio
from typing import Optional
import argparse

from web3 import AsyncWeb3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from config import Config

# Polymarket constants
POLYMARKET_HOST = "https://clob.polymarket.com"

# Conditional Tokens (ERC-1155 positions) on Polygon
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# CTF Exchange on Polygon (v2) - for redemption
CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"

# ABI for Conditional Tokens (ERC-1155)
CONDITIONAL_ABI = [
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
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256", "name": "indexSet", "type": "uint256"}
        ],
        "name": "getCollectionId",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "collectionId", "type": "bytes32"}
        ],
        "name": "getPositionId",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "pure",
        "type": "function"
    }
]

# ABI for CTF Exchange redeem
CTF_EXCHANGE_ABI = [
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

async def transfer_orphaned_position(
    condition_id: str,
    index_set: int,
    amount: float,
    proxy_wallet: str,
    redeem_first: bool = False,
    expected_from: Optional[str] = None,
):
    """Transfer orphaned position tokens to the target wallet.

    If redeem_first=True, attempt to redeem (claim) first. Note redemption sends collateral to
    the caller (your EOA), not the destination wallet.
    """
    print(
        f"Transfer request: amount={amount} shares conditionId={condition_id} indexSet={index_set} to={proxy_wallet} redeem_first={redeem_first}"
    )

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

    if expected_from:
        try:
            expected = w3.to_checksum_address(expected_from)
        except Exception:
            print(f"Error: invalid expected from address: {expected_from}")
            return
        if w3.to_checksum_address(from_address) != expected:
            print(
                "Error: configured private key address does not match expected from address. "
                f"expected={expected} actual={from_address}"
            )
            return

    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=CONDITIONAL_ABI)
    ctf_exchange = w3.eth.contract(address=w3.to_checksum_address(CTF_EXCHANGE), abi=CTF_EXCHANGE_ABI)

    # Polymarket has used both bridged USDC.e and native USDC on Polygon.
    collateral_tokens = [
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC.e
        "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # USDC (native)
    ]
    parent_collection_id = "0x0000000000000000000000000000000000000000000000000000000000000000"

    # Optionally attempt redemption first.
    # NOTE: This pays out collateral to from_address, not proxy_wallet.
    if redeem_first:
        try:
            tx_data = await ctf_exchange.functions.redeemPositions(
                w3.to_bytes(hexstr=condition_id),
                [int(index_set)],
            ).build_transaction({
                "from": from_address,
                "nonce": await w3.eth.get_transaction_count(from_address),
                "gas": 250000,
                "gasPrice": await w3.eth.gas_price,
            })
            signed_tx = account.sign_transaction(tx_data)
            raw = getattr(signed_tx, "raw_transaction", None) or getattr(signed_tx, "rawTransaction")
            tx_hash = await w3.eth.send_raw_transaction(raw)
            print(f"Redeem tx sent! Hash: {tx_hash.hex()}")
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            if receipt.status == 1:
                print("✅ Redeem succeeded (collateral paid to your EOA). No transfer needed.")
                return
            print(f"❌ Redeem failed. Status: {receipt.status}. Will attempt transfer.")
        except Exception as e:
            print(f"Redeem attempt error (continuing to transfer): {e}")

    # Compute ERC-1155 position token id.
    collection_id = None
    position_id = None
    used_collateral = None
    for collateral_token in collateral_tokens:
        try:
            collection_id = await conditional.functions.getCollectionId(
                parent_collection_id,
                w3.to_bytes(hexstr=condition_id),
                index_set,
            ).call()
            position_id = await conditional.functions.getPositionId(
                w3.to_checksum_address(collateral_token),
                collection_id,
            ).call()
            used_collateral = collateral_token
            break
        except Exception:
            continue

    if collection_id is None or position_id is None or used_collateral is None:
        print("Error getting position ID: could not compute position id for either USDC.e or native USDC")
        return

    print(f"Collateral token used: {used_collateral}")
    print(f"Collection ID: {collection_id.hex()}")
    print(f"Position ID: {position_id}")

    # Check balance
    try:
        balance = await conditional.functions.balanceOf(from_address, position_id).call()
        print(f"Current balance: {balance / 1e6:.6f} shares")  # Polymarket position tokens use 6 decimals
        if balance < amount * 1e6:
            print(f"Insufficient balance: have {balance / 1e6:.6f}, need {amount}")
            return
    except Exception as e:
        print(f"Error checking balance: {e}")
        return

    # Transfer
    amount_wei = int(amount * 1e6)  # 6 decimals
    to_address = w3.to_checksum_address(proxy_wallet)

    try:
        tx_data = await conditional.functions.safeTransferFrom(
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

        print("\nTX PREVIEW")
        print(f"  contract: {CONDITIONAL_TOKENS}")
        print(f"  from:     {from_address}")
        print(f"  to:       {to_address}")
        print(f"  tokenId:  {position_id}")
        print(f"  amount:   {amount_wei} (raw; 6 decimals => {amount} shares)")
        print(f"  nonce:    {tx_data.get('nonce')}")
        print(f"  gas:      {tx_data.get('gas')}")
        print(f"  gasPrice: {tx_data.get('gasPrice')}")
        print("\nType YES to broadcast this transaction:")
        if input().strip() != "YES":
            print("Aborted (did not broadcast).")
            return

        # Sign & Send
        signed_tx = account.sign_transaction(tx_data)
        raw = getattr(signed_tx, "raw_transaction", None) or getattr(signed_tx, "rawTransaction")
        tx_hash = await w3.eth.send_raw_transaction(raw)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_id")
    parser.add_argument("index_set", type=int)
    parser.add_argument("amount", type=float)
    parser.add_argument("to_wallet")
    parser.add_argument("--redeem-first", action="store_true", dest="redeem_first")
    parser.add_argument("--from", dest="expected_from", default=None)
    args = parser.parse_args()

    asyncio.run(
        transfer_orphaned_position(
            args.condition_id,
            args.index_set,
            args.amount,
            args.to_wallet,
            redeem_first=args.redeem_first,
            expected_from=args.expected_from,
        )
    )
