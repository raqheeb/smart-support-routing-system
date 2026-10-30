import asyncio
import json
from backend.llm import call_claude

async def main():
    print("Testing call_claude...")
    res = await call_claude("My phone is broken and stopped working!")
    print("Result:", json.dumps(res, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
