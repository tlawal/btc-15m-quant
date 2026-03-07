import asyncio
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from config import Config
import json

w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL))
try:
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
except Exception:
    pass

address = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
print(f"Address: {address}")

GNOSIS_CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
GNOSIS_ABI = [
    {
        "inputs": [
            {"internalType": "contract IERC20", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

ctf = w3.eth.contract(address=w3.to_checksum_address(GNOSIS_CTF), abi=GNOSIS_ABI)
condition_id = "0x149a361ba8d37fe6ad6971afbd8e2aabd36d815a8da42e2c4be3d80ceb03aa3d"

parent_col = b'\x00' * 32
usdc_e = w3.to_checksum_address(Config.POLYGON_USDC_ADDRESS) # "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

try:
    # Build
    tx = ctf.functions.redeemPositions(
        usdc_e,
        parent_col,
        w3.to_bytes(hexstr=condition_id),
        [1, 2] # binary YES/NO
    ).build_transaction({
        'from': address,
        'nonce': w3.eth.get_transaction_count(address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })
    print("Build tx suceeded!")
    
    # Simulate
    ctf.functions.redeemPositions(
        usdc_e,
        parent_col,
        w3.to_bytes(hexstr=condition_id),
        [1, 2]
    ).call({'from': address})
    
    print("Simulated call suceeded!")
    
    # If successful, actually execute it to see if balance increases!
    signed = w3.eth.account.sign_transaction(tx, private_key=Config.POLYMARKET_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Sent tx: {tx_hash.hex()}")
    
except Exception as e:
    print(f"Error: {e}")
