#!/usr/bin/env sh
set -eu

docker run --rm \
  -v "$PWD:/opt/metarank" \
  metarank/metarank:latest \
  validate \
  --config /opt/metarank/config.yml \
  --data /opt/metarank/events.jsonl
