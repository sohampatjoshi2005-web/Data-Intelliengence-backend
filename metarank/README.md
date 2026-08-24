# MetaRank Bootstrap

This folder contains a first-pass MetaRank setup for recommendations built from:

- `/Users/sathya/Downloads/customers_5000.json`

## Files
- `config.yml`: MetaRank model config
- `events.jsonl`: generated bootstrap events (created by the build script)

## Models
- `similar-products`: collaborative filtering with ALS
- `trending-products`: weighted popularity fallback

## Build events
```bash
cd "/Users/sathya/Downloads/Agentic-auto-ml-main 2"
python scripts/build_metarank_events.py --input /Users/sathya/Downloads/customers_5000.json --output metarank/events.jsonl --views
```

## Start MetaRank in Docker
Official docs show a standalone Docker workflow using a local config file and data file.

```bash
cd "/Users/sathya/Downloads/Agentic-auto-ml-main 2"
docker run -i -t -p 8080:8080 \
  -v "$PWD/metarank:/opt/metarank" \
  metarank/metarank:latest \
  standalone --config /opt/metarank/config.yml --data /opt/metarank/events.jsonl
```

## Alternative: live load over /feedback
If MetaRank is already running separately:

```bash
cd "/Users/sathya/Downloads/Agentic-auto-ml-main 2"
python scripts/load_metarank_events.py --events metarank/events.jsonl --url http://127.0.0.1:8080/feedback
```

## Test recommendations
Trending:
```bash
python scripts/test_metarank_recommend.py --model trending-products --count 5
```

Similar items:
```bash
python scripts/test_metarank_recommend.py --model similar-products --item P1001 --count 5
```

Replace `P1001` with a real `Product_ID` from the dataset.

## Notes
- this dataset is strong enough to bootstrap recommendations
- it is **not** a true ranking/impression dataset yet
- later we should add live `ranking` and `interaction` events for reranking/LTR
