# test_credentials.py — run this to verify EOA credentials work
import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv()  # loads your local .env (or Railway will use env vars)

private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
funder = os.getenv("FUNDER_ADDRESS")

if not private_key:
    print("❌ Missing POLYMARKET_PRIVATE_KEY in .env or env vars")
    exit(1)

print("✅ Private key found — attempting EOA credential derivation...")

client = ClobClient(
    host="https://clob.polymarket.com",
    key=private_key,           # ← your EOA private key
    chain_id=137,
    funder=funder or None
)

# This is the exact line that fixes everything
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)

print("✅ SUCCESS: Credentials derived and set!")
print("API Key:", creds.api_key[:8] + "..." if creds.api_key else "None")
print("You can now place orders with this client.")