import asyncio
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from polymarket_client import PolymarketClient


def test_limit_sell_preserves_fractional_size_to_4dp():
    async def _run():
        # Bypass __init__ (would try to derive creds / connect)
        pm = PolymarketClient.__new__(PolymarketClient)
        pm.can_trade = True

        async def _ok_allowance(*args, **kwargs):
            return True

        pm._ensure_conditional_allowance = _ok_allowance

        captured = {}

        class _DummyClient:
            def create_order(self, args):
                captured["size"] = float(getattr(args, "size", None))
                return {"signed": True}

            def post_order(self, signed, ot):
                return {"orderID": "0xdeadbeef"}

            def update_balance_allowance(self, params):
                return None

        pm.client = _DummyClient()

        oid = await pm.limit_sell("token", price=0.98, size=6.99444, order_type="FOK")
        assert oid == "0xdeadbeef"
        # floor to 4 decimals
        assert captured["size"] == 6.9944

    asyncio.run(_run())
