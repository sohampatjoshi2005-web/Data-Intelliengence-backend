# AutoML Backend - Bedrock Claude Haiku Migration Summary

## Overview
Migrated the AutoML backend from Ollama (local LLM) to AWS Bedrock Claude Haiku for production deployment on EC2.

## Key Changes

### 1. Environment Configuration (.env.ec2)
**Changed from Ollama to Bedrock:**
```env
# OLD
STRUCTURED_LLM_PROVIDER=ollama_local
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# NEW
STRUCTURED_LLM_PROVIDER=bedrock
BEDROCK_REGION=us-east-2
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_MAX_TOKENS=2048
BEDROCK_TEMPERATURE=0.3
```

### 2. LLM Router (app/core/llm_clients.py)
**Updated routing priority:**
- Primary: AWS Bedrock (Claude Haiku)
- Fallback: Ollama (if configured)

**Enhancements:**
- Bedrock is now the primary provider (not fallback)
- Added `invoke_model_with_response_stream()` support for streaming responses
- Improved error handling and provider availability detection
- Streaming now works with Bedrock using proper event parsing

### 3. Configuration Defaults (app/core/config.py)
**Changed default LLM provider:**
```python
# OLD
structured_llm_provider: str = os.getenv("STRUCTURED_LLM_PROVIDER", "ollama_local")

# NEW
structured_llm_provider: str = os.getenv("STRUCTURED_LLM_PROVIDER", "bedrock")
```

### 4. Main Application (app/main.py)
**Updated defaults and helper function:**
- Changed `_structured_provider()` to recognize both "bedrock" and "ollama_local" as defaults
- Updated `/models` endpoint to report "bedrock" as default provider
- Changed `_run_structured_pipeline()` default parameter from "ollama_local" to "bedrock"

### 5. Task Queue (app/tasks.py)
**Updated Celery task defaults:**
- `orchestrate_pipeline()` now defaults to "bedrock"
- Updated provider resolution logic to fall back to configured `STRUCTURED_LLM_PROVIDER`

### 6. CORS Configuration
**Added EC2 endpoints:**
```env
CORS_ORIGINS=...http://127.0.0.1:8000,http://3.135.222.192:8000,...
```

## Deployment Requirements

### EC2 Prerequisites
1. **IAM Role:** EC2 instance must have `AmazonBedrockFullAccess` policy attached
2. **Bedrock Access:** Enable Claude Haiku in Bedrock Console → Model Access
3. **AWS Region:** us-east-2 (ensure EC2 is in same region)
4. **Environment Variables:** Ensure `.env.ec2` is loaded on deployment

### Cost Implications
- **Claude Haiku Input:** $0.00025 / 1K tokens
- **Claude Haiku Output:** $0.00125 / 1K tokens
- **Estimated Cost:** 1M input tokens ≈ $0.25 (very cost-effective for testing)

## Alternative Model IDs
If newer Claude Haiku not available:
```env
# Older, cheaper variant
BEDROCK_MODEL_ID=anthropic.claude-haiku-20240307-v1:0
```

## Testing the Configuration

### 1. Verify Bedrock Access
```python
import boto3
client = boto3.client("bedrock-runtime", region_name="us-east-2")
response = client.invoke_model(
    modelId="anthropic.claude-haiku-4-5-20251001-v1:0",
    body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}),
    contentType="application/json"
)
```

### 2. Start Backend
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Check Health Endpoint
```bash
curl http://127.0.0.1:8000/health
```

Should show:
```json
{
  "status": "healthy",
  "providers": ["bedrock", "ollama_local"],
  "default_provider": "bedrock",
  "bedrock_model": "anthropic.claude-haiku-4-5-20251001-v1:0"
}
```

## Rollback Plan
If issues occur with Bedrock:
1. Set `STRUCTURED_LLM_PROVIDER=ollama_local` in .env
2. Ensure Ollama is running on port 11434
3. Restart backend service

## Files Modified
1. ✓ backend/.env.ec2
2. ✓ backend/app/core/llm_clients.py
3. ✓ backend/app/core/config.py
4. ✓ backend/app/main.py
5. ✓ backend/app/tasks.py

## Next Steps
1. Compress backend: `tar -czf automl-backend.tar.gz backend/`
2. Upload to EC2 via SCP
3. Deploy and configure systemd service
4. Verify Bedrock connectivity
5. Monitor logs for any LLM errors
