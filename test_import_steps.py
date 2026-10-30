import sys
print("Step 1: start", flush=True)
import json
print("Step 2: json ok", flush=True)
import httpx
print("Step 3: httpx ok", flush=True)
from backend.config import ANTHROPIC_API_KEY
print("Step 4: config ok", flush=True)
from backend.llm import call_claude
print("Step 5: llm ok", flush=True)

import asyncio
print("Step 6: running call_claude...", flush=True)
res = asyncio.run(call_claude("My phone is broken and stopped working!"))
print("Step 7: done!", res, flush=True)
