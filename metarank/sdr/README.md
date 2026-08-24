# SDR MetaRank

This configuration and event stream are isolated from the repository's product
recommendation models.

## Event lifecycle

1. Prospect Intelligence writes a `ranking` event for every candidate queue.
2. Outreach and Conversation write later interaction events such as `opened`,
   `clicked`, `replied`, `interested`, `pricing_request`, `qualified`,
   `meeting_request`, `bounced`, and `unsubscribe`.
3. Cold start uses the local probability-weighted ranking.
4. After sufficient ranking groups and labeled outcomes exist, train the
   `sdr-prospect-ranker` LambdaMART model and configure:

```bash
export SDR_INTELLIGENCE_METARANK_URL=http://127.0.0.1:8081
export SDR_INTELLIGENCE_METARANK_MODEL=sdr-prospect-ranker
export SDR_INTELLIGENCE_METARANK_EVENT_PATH=metarank/sdr/events.jsonl
```

The application status endpoint reports whether learned reranking is reachable:

```bash
curl http://127.0.0.1:8000/prospect-intelligence/ranking/status
```

Do not claim the model is trained merely because the YAML exists. MetaRank must
be started with this config and enough valid SDR ranking/outcome events must be
available for training.

## Validate and start

The official MetaRank v0.7.11 validator passes for `config.yml` and
`seed-events.jsonl`.

After the application has collected real events at `events.jsonl`:

```bash
cd metarank/sdr
docker run --rm \
  -v "$PWD:/opt/metarank" \
  metarank/metarank:latest \
  validate --config /opt/metarank/config.yml --data /opt/metarank/events.jsonl

docker compose up -d
```

`standalone` imports the event stream, trains the configured model, and serves
the ranking API on port `8081`. Do not start production training with only the
synthetic seed file.
