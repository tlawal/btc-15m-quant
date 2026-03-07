import asyncio
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from config import Config
import json

w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL))
try:
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
except:
    pass

address = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
cond_id = "0x149a361bd4e2eebe4fdb2c7a33116debe6ff75c1fa4189255ea30bd4c1800ae7"
parent_col = "0x0000000000000000000000000000000000000000000000000000000000000000"

CTF = w3.eth.contract(address="0x4D97DCd97eC945f40cF65F87097ACe5EA0476045", abi=[{
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
        "inputs": [{"internalType": "bytes32", "name": "conditionId", "type": "bytes32"}],
        "name": "payoutNumerators",
        "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
])


usdc_e = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
usdc = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

for tk in [usdc_e, usdc]:
    col1 = CTF.functions.getCollectionId(tk, parent_col, cond_id, 1).call()
    col2 = CTF.functions.getCollectionId(tk, parent_col, cond_id, 2).call()
    
    pos1 = int(col1.hex() + "0"*24, 16) % (2**256) # CTF positions are collectionId + some padding? Wait, positionId formula:
    
    print(f"Token {tk}: Collection YES={col1.hex()} NO={col2.hex()}")
    
try:
    payout = CTF.functions.payoutNumerators(cond_id).call()
    print(f"Payout numerators: {payout}")
except Exception as e:
    print(f"Payout numerators not found: {e}")
