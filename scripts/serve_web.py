#!/usr/bin/env python3
"""Serve the local MathCoach web application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathcoach.web.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve MathCoach local web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes during local development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"MathCoach Web: http://{args.host}:{args.port}", flush=True)
    if args.reload:
        uvicorn.run(
            "mathcoach.web.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
