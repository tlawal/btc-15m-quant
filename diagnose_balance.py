"""
Diagnostic script: Check USDC balance at every layer of the pipeline.

Traces:
  1. EOA address derived from private key
  2. USDC balance of the EOA (direct on-chain balanceOf)
  3. Polymarket proxy wallet address (deterministic, derived from EOA)
  4. USDC balance of the proxy wallet
  5. CLOB API get_balance_allowance result (raw)
  6. Parsed balance values

Run: python diagnose_balance.py
"""

import asyncio
import os
import json
import aiohttp
from dotenv import load_dotenv
from eth_account import Account

load_dotenv()

POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", "")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYGON_USDC_ADDRESS = os.getenv("POLYGON_USDC_ADDRESS", "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359")

# Polymarket CTF Exchange proxy factory on Polygon
# The proxy wallet = CREATE2(factory, salt=eoa_address, init_code)
# We can compute it, but easier to query the CLOB API for it.
POLYMARKET_PROXY_FACTORY = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Polymarket also uses USDC.e (bridged) in addition to native USDC
POLYGON_USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (bridged, 6 decimals)


def pad32(hex_no_0x: str) -> str:
    return hex_no_0x.rjust(64, "0")


async def check_erc20_balance(session: aiohttp.ClientSession, rpc_url: str, token: str, wallet: str, label: str):
    """Call balanceOf(wallet) on an ERC20 token via eth_call."""
    owner = wallet.lower().replace("0x", "")
    data = "0x70a08231" + pad32(owner)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": token.lower(), "data": data}, "latest"],
    }
    try:
        async with session.post(rpc_url, json=payload) as r:
            resp = await r.json()
        raw_hex = resp.get("result", "0x0")
        if not raw_hex or raw_hex == "0x":
            raw_hex = "0x0"
        raw_int = int(raw_hex, 16)
        usdc = raw_int / 1_000_000
        print(f"  {label}: raw={raw_int} → ${usdc:.6f} USDC")
        return usdc
    except Exception as e:
        print(f"  {label}: ERROR: {e}")
        return 0.0


async def check_clob_balance():
    """Check balance via py-clob-client get_balance_allowance."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams

        secret = os.getenv("POLYMARKET_API_SECRET", "")
        if secret and len(secret) % 4 != 0:
            secret += "=" * (4 - (len(secret) % 4))

        client = ClobClient(
            host="https://clob.polymarket.com",
            key=POLYMARKET_PRIVATE_KEY,
            chain_id=137,
            creds=ApiCreds(
                api_key=os.getenv("POLYMARKET_API_KEY", ""),
                api_secret=secret,
                api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
            ),
        )
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = client.get_balance_allowance(params)
        print(f"\n3. CLOB get_balance_allowance (raw):")
        print(f"   Type: {type(result)}")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"   {k}: {v} (type={type(v).__name__})")
                try:
                    val = float(v)
                    if val >= 1e5:
                        print(f"      → parsed as micro-USDC: ${val / 1e6:.6f}")
                    else:
                        print(f"      → parsed as USDC: ${val:.6f}")
                except:
                    pass
        else:
            print(f"   Value: {result}")
        return result
    except Exception as e:
        print(f"\n3. CLOB get_balance_allowance: ERROR: {e}")
        return None


async def get_proxy_wallet(session: aiohttp.ClientSession, eoa: str):
    """Try to get the Polymarket proxy wallet address via the CLOB API."""
    try:
        url = f"https://clob.polymarket.com/profile?address={eoa}"
        async with session.get(url) as r:
            if r.status == 200:
                data = await r.json()
                proxy = data.get("proxyWallet") or data.get("proxy_wallet") or data.get("proxyAddress")
                if proxy:
                    return proxy
    except:
        pass

    # Alternative: try derive_api_key endpoint
    try:
        url = f"https://clob.polymarket.com/auth/derive-api-key"
        # This needs auth, skip
    except:
        pass

    return None


async def main():
    print("=" * 60)
    print("  BTC 15m Quant — USDC Balance Diagnostic")
    print("=" * 60)

    # 1. Derive EOA
    pk = POLYMARKET_PRIVATE_KEY
    if not pk:
        print("\n❌ POLYMARKET_PRIVATE_KEY not set!")
        return

    # Ensure 0x prefix
    if not pk.startswith("0x"):
        pk = "0x" + pk

    acct = Account.from_key(pk)
    eoa = acct.address
    print(f"\n1. EOA Address: {eoa}")

    # 2. Check RPC
    if not POLYGON_RPC_URL:
        print("\n❌ POLYGON_RPC_URL not set! Cannot check on-chain balances.")
        print("   Set POLYGON_RPC_URL in .env (e.g., https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY)")
        # Still try CLOB
        await check_clob_balance()
        return

    print(f"   RPC: {POLYGON_RPC_URL[:50]}...")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        # 2a. Check MATIC/POL balance (for gas)
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_getBalance",
                "params": [eoa, "latest"],
            }
            async with session.post(POLYGON_RPC_URL, json=payload) as r:
                resp = await r.json()
            raw = int(resp.get("result", "0x0"), 16)
            pol = raw / 1e18
            print(f"   POL/MATIC: {pol:.6f}")
        except Exception as e:
            print(f"   POL/MATIC: ERROR: {e}")

        # 2b. Check USDC balances on EOA
        print(f"\n2. On-chain USDC balances for EOA ({eoa}):")
        eoa_native = await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_ADDRESS, eoa, "USDC (native)")
        eoa_bridged = await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_E_ADDRESS, eoa, "USDC.e (bridged)")

        # 3. Check CLOB balance
        clob_result = await check_clob_balance()

        # 4. Try to find proxy wallet
        print(f"\n4. Polymarket Proxy Wallet:")
        proxy = await get_proxy_wallet(session, eoa)
        if proxy:
            print(f"   Proxy address: {proxy}")
            print(f"   Checking USDC balances on proxy wallet:")
            proxy_native = await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_ADDRESS, proxy, "USDC (native)")
            proxy_bridged = await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_E_ADDRESS, proxy, "USDC.e (bridged)")
        else:
            print("   Could not determine proxy wallet address via API.")
            print("   Check https://polymarket.com with your wallet to find it.")

            # Try common proxy factory pattern
            # Polymarket proxy wallets are often at a deterministic address
            print("\n   Trying known proxy patterns...")
            # The proxy is created by the Polymarket proxy factory
            # We can try to query the factory for the proxy address
            try:
                # gnosis safe proxy: getProxy(owner) on factory
                # Polymarket uses a custom factory at POLYMARKET_PROXY_FACTORY
                owner_padded = pad32(eoa.lower().replace("0x", ""))
                # Try getProxyFor(address) selector: different factories use different selectors
                for selector in ["0x5c60da1b", "0x6e296e45"]:  # implementation(), getAddress()
                    data = selector + owner_padded
                    payload = {
                        "jsonrpc": "2.0", "id": 1,
                        "method": "eth_call",
                        "params": [{"to": POLYMARKET_PROXY_FACTORY, "data": data}, "latest"],
                    }
                    async with session.post(POLYGON_RPC_URL, json=payload) as r:
                        resp = await r.json()
                    result = resp.get("result", "0x")
                    if result and result != "0x" and len(result) >= 42:
                        addr = "0x" + result[-40:]
                        if int(addr, 16) != 0:
                            print(f"   Potential proxy: {addr}")
                            await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_ADDRESS, addr, "USDC (native)")
                            await check_erc20_balance(session, POLYGON_RPC_URL, POLYGON_USDC_E_ADDRESS, addr, "USDC.e (bridged)")
                            break
            except Exception as e:
                print(f"   Factory query error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)
    if eoa_native > 0 or eoa_bridged > 0:
        total = eoa_native + eoa_bridged
        print(f"  ✅ EOA has ${total:.2f} USDC")
        print(f"     The on-chain balance check SHOULD work.")
        print(f"     If bot still shows $0, check that get_wallet_usdc_balance()")
        print(f"     is using the correct USDC contract address.")
    else:
        print(f"  ⚠️  EOA has $0 USDC")
        print(f"     Your USDC is likely in the Polymarket PROXY wallet,")
        print(f"     not the EOA. The bot needs to check the proxy wallet")
        print(f"     address instead.")
        print()
        print(f"  💡 FIX: The bot should query the CLOB API for the balance,")
        print(f"     or we need to find your proxy wallet address.")
        if clob_result and isinstance(clob_result, dict):
            bal = clob_result.get("balance", "?")
            print(f"     CLOB reports balance={bal}")


if __name__ == "__main__":
    asyncio.run(main())
