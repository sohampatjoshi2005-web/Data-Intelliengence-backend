#!/usr/bin/env python3
"""Load MetaRank events into the /feedback endpoint in batches."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Iterable, List, Dict, Any


DEFAULT_EVENTS = "metarank/events.jsonl"
DEFAULT_URL = "http://127.0.0.1:8080/feedback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--batch-size", type=int, default=250)
    return parser.parse_args()


def chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def post_json(url: str, payload: Any) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    events_path = Path(args.events)
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]

    accepted = 0
    for batch in chunks(events, args.batch_size):
        result = post_json(args.url, batch)
        accepted += int(result.get("accepted", 0))
        print(result)

    print(json.dumps({"submitted": len(events), "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
