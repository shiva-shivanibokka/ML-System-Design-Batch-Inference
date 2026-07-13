"""
Vercel entrypoint for the serving API.
======================================
Thin shim: Vercel's Python runtime builds every file under /api into a serverless
function, so this single file re-exports the FastAPI app. `vercel.json` rewrites
all routes here, so the whole API (all routes, /docs, /health) is served by one
function. The real app lives in serving/ (a distinctively-named package so it
never collides with other repos' `api` packages on a shared PYTHONPATH).
"""

import sys
from pathlib import Path

# Repo root on the path so `serving`, `configs`, `db` import cleanly on Vercel.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Vercel's @vercel/python detects and serves the ASGI `app`.
