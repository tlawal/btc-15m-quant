import asyncio
from web3 import Web3
try:
    # web3 < 6
    from web3.middleware import geth_poa_middleware  # type: ignore
except Exception:
    try:
        # web3 >= 6
        from web3.middleware.geth_poa import geth_poa_middleware  # type: ignore
    except Exception:
        # web3 >= 7
        from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware  # type: ignore

        def geth_poa_middleware(make_request, w3):
            return ExtraDataToPOAMiddleware(make_request, w3)
from config import Config

w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

address = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY).address

CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"
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
ctf = w3.eth.contract(address=w3.to_checksum_address(CTF_EXCHANGE), abi=CTF_ABI)
condition_id = "0x149a361bd4e2eebe4fdb2c7a33116debe6ff75c1fa4189255ea30bd4c1800ae7"

try:
    print("Calling call()...")
    ctf.functions.redeemPositions(
        Web3.to_bytes(hexstr=condition_id),
        [1, 2]
    ).call({'from': address})
    print("Simulated call successful!")
except Exception as e:
    print(f"Simulation error: {e}")
