#!/usr/bin/env python3
"""Exercise MetaRank recommend endpoints for the bootstrap models."""

from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.error
from typing import Any, Dict


DEFAULT_BASE = "http://127.0.0.1:8080"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--model", choices=["similar-products", "trending-products"], required=True)
    parser.add_argument("--item", help="Context item id for similar-products, e.g. P1001")
    parser.add_argument("--user", help="Optional user id")
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"MetaRank request failed with HTTP {exc.code}: {body}") from exc


def main() -> None:
    args = parse_args()
    payload: Dict[str, Any] = {"count": args.count}
    if args.user:
        payload["user"] = args.user
    if args.item:
        # MetaRank similar-item recommendations expect raw item ids here.
        payload["items"] = [args.item]

    url = f"{args.base_url}/recommend/{args.model}"
    result = post_json(url, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
