import asyncio
import httpx
from config import Config

async def get_condition_id_from_slug(slug: str):
    """Get condition ID from Polymarket market slug."""
    url = f"https://gamma-api.polymarket.com/markets/{slug}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            condition_id = data.get("conditionId")
            if condition_id:
                print(f"Condition ID for {slug}: {condition_id}")
                return condition_id
            else:
                print("Condition ID not found in market data")
        except Exception as e:
            print(f"Error fetching market data: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python get_condition_from_slug.py <slug>")
        sys.exit(1)

    slug = sys.argv[1]
    asyncio.run(get_condition_id_from_slug(slug))
