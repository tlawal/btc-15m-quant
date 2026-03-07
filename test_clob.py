from py_clob_client.client import ClobClient, ApiCreds
c = ClobClient(host="https://clob.polymarket.com", key="0x" + "1"*64, chain_id=137)
print("has web3?", hasattr(c, "web3"))
print("what is it?", getattr(c, "web3", None))
