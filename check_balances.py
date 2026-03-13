import asyncio
from web3 import AsyncWeb3
from config import Config

# Addresses (Polygon mainnet)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ERC-1155 balanceOf signature
ERC1155_BALANCE_OF = "0x00fdd58e"  # balanceOf(address, uint256)

async def main():
    if not Config.POLYGON_RPC_URL:
        print("Error: POLYGON_RPC_URL not set in config.py")
        return

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
    if not await w3.is_connected():
        print("Error: Could not connect to Polygon RPC.")
        return

    account = w3.to_checksum_address("0x7AbA1F81034d418A4DED1613626cA7573FD85153")

    # 1. Check USDC balance (6 decimals)
    usdc_contract = w3.eth.contract(address=w3.to_checksum_address(USDC_ADDRESS), abi=[
        {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"}
    ])
    usdc_balance_raw = await usdc_contract.functions.balanceOf(account).call()
    usdc_balance = usdc_balance_raw / 1_000_000  # 6 decimals
    print(f"USDC balance in EOA: ${usdc_balance:.6f}")

    # 2. Check ERC-1155 balance for the YES token of that condition
    # Token IDs: from your market's clobTokenIds (YES is the first one)
    yes_token_id = 104396440751277512071663049096787928567222444925008417395090357720411752196622
    no_token_id = 11938953172475957203811815164637302188981958345642701087732431672495508349666

    conditional = w3.eth.contract(address=w3.to_checksum_address(CONDITIONAL_TOKENS), abi=[
        {"constant": True, "inputs": [{"name": "account", "type": "address"}, {"name": "id", "type": "uint256"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
    ])

    yes_balance = await conditional.functions.balanceOf(account, yes_token_id).call()
    no_balance = await conditional.functions.balanceOf(account, no_token_id).call()
    print(f"ERC-1155 YES balance: {yes_balance}")
    print(f"ERC-1155 NO balance: {no_balance}")

if __name__ == "__main__":
    asyncio.run(main())
