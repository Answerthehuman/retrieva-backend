import asyncio
import os
import sys

# Ensure backend directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rag.ingestion.service import IngestionService
from api.main import app

async def test():
    print("Imports successful!")
    print(f"App routes: {[r.path for r in app.routes]}")

if __name__ == "__main__":
    asyncio.run(test())
