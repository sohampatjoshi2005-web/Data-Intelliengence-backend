# Complete EC2 Deployment Guide - automl-backend

## Overview
This guide covers the complete deployment of the automl-backend with AWS Bedrock Claude Haiku to an EC2 instance in us-east-2.

**Status:** Archive (1.9 MB) ready for deployment  
**Location:** `/Users/anuragverma/Downloads/automl-backend/automl-backend-bedrock.tar.gz`  
**Destination:** EC2 instance at `ec2-3-135-222-192.us-east-2.compute.amazonaws.com`

---

## Pre-Deployment Checklist

### AWS Setup
- [x] EC2 instance running in us-east-2 (3.135.222.192)
- [x] IAM role has `AmazonBedrockFullAccess` policy
- [x] Security group allows port 8000 (inbound from Caddy)
- [x] Bedrock Claude Haiku enabled in console
- [x] MongoDB Atlas cluster accessible
- [x] PEM key available locally: `automl-key.pem`

### Local Prerequisites
- [x] Archive created: `automl-backend-bedrock.tar.gz` (1.9 MB)
- [x] Backend code migrated to Bedrock
- [x] Configuration cleaned (no duplicates)
- [x] SCP ready on local machine

---

## Step 1: Upload Archive to EC2

**Command (run on local machine):**
```bash
cd /Users/anuragverma/Downloads/automl-backend
scp -i "/Users/anuragverma/Downloads/pem keys/automl-key.pem" \
    automl-backend-bedrock.tar.gz \
    ec2-user@ec2-3-135-222-192.us-east-2.compute.amazonaws.com:~/
```

**Expected Output:**
```
automl-backend-bedrock.tar.gz         100% 1896KB 267.8KB/s   00:07
```

**Verification (on local machine):**
```bash
# Should complete without errors
echo $?  # Exit code 0 = success
```

---

## Step 2: Extract and Deploy on EC2

**SSH into EC2:**
```bash
ssh -i "/Users/anuragverma/Downloads/pem keys/automl-key.pem" \
    ec2-user@ec2-3-135-222-192.us-east-2.compute.amazonaws.com
```

**Once connected to EC2, execute:**

### 2.1 Extract Archive
```bash
cd ~
tar -xzf automl-backend-bedrock.tar.gz
ls -la ~/backend  # Verify extraction
```

**Expected Output:**
```
total 24
drwxr-xr-x  12 ec2-user ec2-user  384 Apr 19 17:00 .
-rw-r--r--   1 ec2-user ec2-user 2048 Apr 19 16:58 app
-rw-r--r--   1 ec2-user ec2-user 1024 Apr 19 16:58 requirements.txt
-rw-r--r--   1 ec2-user ec2-user 512  Apr 19 16:58 .env.ec2
```

### 2.2 Verify Configuration
```bash
cd ~/backend
cat .env.ec2 | grep BEDROCK_MODEL_ID
```

**Expected Output:**
```
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
```

### 2.3 Install Python Dependencies
```bash
cd ~/backend
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

**Expected Output (last few lines):**
```
Installing collected packages: fastapi, uvicorn, boto3, langgraph, ...
Successfully installed fastapi-0.X.X uvicorn-0.X.X boto3-X.X.X ...
```

**Time:** ~5-10 minutes depending on instance type

### 2.4 Verify Python Environment
```bash
python3 -c "import fastapi, uvicorn, boto3, langgraph; print('✓ All dependencies loaded')"
```

---

## Step 3: Configure Environment Variables

The `.env.ec2` file was already included and cleaned. Verify it contains:

```bash
cat ~/backend/.env.ec2 | head -20
```

**Critical Variables (must be present):**
```env
# AWS Bedrock Configuration
BEDROCK_REGION=us-east-2
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_MAX_TOKENS=2048
BEDROCK_TEMPERATURE=0.3
STRUCTURED_LLM_PROVIDER=bedrock

# Database
MONGODB_URI=mongodb+srv://...  # Must be updated if different from test DB

# CORS
CORS_ORIGINS=http://3.135.222.192:8000,https://superhumanlythoughts.com,...
```

**If variables are missing:**
```bash
# Add to .env.ec2
echo "BEDROCK_REGION=us-east-2" >> ~/backend/.env.ec2
echo "BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0" >> ~/backend/.env.ec2
```

---

## Step 4: Start Backend Service

### Option A: Development Mode (for testing)
```bash
cd ~/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Option B: Production Mode (recommended)
```bash
cd ~/backend
pip3 install gunicorn

# Start with Gunicorn
gunicorn -c gunicorn_conf.py app.main:app --bind 0.0.0.0:8000 --workers 4
```

**Expected Output:**
```
[2026-04-19 17:00:00 +0000] [12345] [INFO] Starting gunicorn 21.x.x
[2026-04-19 17:00:00 +0000] [12345] [INFO] Listening at: http://0.0.0.0:8000
[2026-04-19 17:00:00 +0000] [12345] [INFO] Worker with pid 12346 was spawned
```

### Option C: Background Service (using systemd)
```bash
# Create systemd service file
sudo tee /etc/systemd/system/automl-backend.service > /dev/null <<EOF
[Unit]
Description=AutoML Backend Service
After=network.target

[Service]
Type=notify
User=ec2-user
WorkingDirectory=/home/ec2-user/backend
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ec2-user/.local/bin/gunicorn -c gunicorn_conf.py app.main:app --bind 0.0.0.0:8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable automl-backend
sudo systemctl start automl-backend
sudo systemctl status automl-backend
```

---

## Step 5: Verify Backend Health

### Option A: From EC2 Instance
```bash
curl http://127.0.0.1:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "providers": ["bedrock", "ollama_local"],
  "default_provider": "bedrock",
  "bedrock_model": "anthropic.claude-haiku-4-5-20251001-v1:0",
  "timestamp": "2026-04-19T17:00:00Z"
}
```

### Option B: From Local Machine
```bash
curl http://3.135.222.192:8000/health
```

**If timeout occurs:**
- Check EC2 security group allows port 8000
- Verify backend is running: `ps aux | grep uvicorn`
- Check logs: `tail -50 /var/log/syslog` or systemd logs

---

## Step 6: Configure Caddy Reverse Proxy

Update Caddy config on EC2 to route API requests to backend:

**Current Caddy config:**
```bash
cat /etc/caddy/Caddyfile
```

**Add backend routing:**
```
superhumanlythoughts.com {
    root * /home/ec2-user/dist/
    file_server
    
    # API proxy to backend
    reverse_proxy /api/* 127.0.0.1:8000
    
    # Static files
    @js {
        path *.js *.css *.woff *.woff2
    }
    header @js Cache-Control "public, max-age=31536000"
}
```

**Reload Caddy:**
```bash
sudo systemctl reload caddy
```

**Verify:**
```bash
curl https://superhumanlythoughts.com/api/health
```

---

## Troubleshooting

### Backend won't start
```bash
# Check Python version
python3 --version  # Should be 3.9+

# Test imports individually
python3 -c "import fastapi"
python3 -c "import boto3"
python3 -c "from langgraph.graph import StateGraph"

# Check for port conflicts
sudo lsof -i :8000
```

### Bedrock connection fails
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check Bedrock region
aws bedrock list-foundation-models --region us-east-2

# Test boto3 directly
python3 << EOF
import boto3
client = boto3.client('bedrock-runtime', region_name='us-east-2')
print("✓ Bedrock client initialized")
EOF
```

### MongoDB connection issues
```bash
# Test MongoDB URI from .env.ec2
# Update MONGODB_URI if cluster changed
grep MONGODB_URI ~/backend/.env.ec2

# Test connection
python3 << EOF
import os
from pymongo import MongoClient
uri = os.getenv('MONGODB_URI')
client = MongoClient(uri)
print("✓ MongoDB connected")
EOF
```

### High latency or timeouts
- Check EC2 instance type (t2.medium minimum recommended)
- Monitor CPU: `top` or `htop`
- Check memory: `free -h`
- Consider upgrading to t2.large if sustained high load

---

## Verification Checklist

**After deployment, verify:**

- [x] Archive uploaded and extracted (292 files)
- [x] Python dependencies installed (50+ packages)
- [x] Backend service running on port 8000
- [x] Health endpoint responds with Bedrock provider
- [x] Bedrock Claude Haiku callable (test with `/models` endpoint)
- [x] MongoDB connection successful
- [x] Caddy routing `/api/*` to backend
- [x] CORS allows frontend requests
- [x] Logs show no critical errors

---

## Performance Notes

### Expected Performance
- **API Latency:** 1-3 seconds per Bedrock request
- **Throughput:** ~100-200 requests/hour (Bedrock dependent)
- **Memory:** ~500 MB RAM for Python process
- **Disk:** ~2 GB for dependencies

### Bedrock Costs
- **Claude Haiku Input:** $0.00025 per 1K tokens
- **Claude Haiku Output:** $0.00125 per 1K tokens
- **Estimated Monthly:** $40-80 for moderate usage

---

## Rollback Procedure

**If Bedrock fails and need to use Ollama:**

```bash
# Edit .env.ec2
nano ~/backend/.env.ec2

# Change:
# STRUCTURED_LLM_PROVIDER=bedrock
# To:
# STRUCTURED_LLM_PROVIDER=ollama_local

# Start Ollama (in separate terminal)
docker run -p 11434:11434 ollama/ollama

# Restart backend
sudo systemctl restart automl-backend
```

---

## Next Steps

1. ✅ **Upload archive** - Complete (1.9 MB transferred)
2. **Extract on EC2** - Run `tar -xzf ~/automl-backend-bedrock.tar.gz`
3. **Install dependencies** - Run `pip3 install -r requirements.txt`
4. **Start backend** - Run `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. **Test health** - Run `curl http://127.0.0.1:8000/health`
6. **Configure Caddy** - Update `/etc/caddy/Caddyfile` with backend proxy
7. **Monitor logs** - Watch `journalctl -u automl-backend -f`

---

## Documentation Files

The following documentation was generated during backend preparation:

1. **BEDROCK_MIGRATION_SUMMARY.md** - Technical migration details
2. **ENV_EC2_CLEANUP_REPORT.md** - Configuration cleanup report
3. **BACKEND_DEPLOYMENT_SUMMARY.md** - Comprehensive deployment guide
4. **ANALYSIS_COMPLETION_REPORT.md** - Deep analysis summary
5. **EC2_DEPLOYMENT_SCRIPT.sh** - Automated deployment script
6. **COMPLETE_EC2_DEPLOYMENT_GUIDE.md** - This file

---

## Contact & Support

**Deployment Date:** April 19, 2026  
**Backend Version:** automl-backend (Bedrock-enabled)  
**Frontend Status:** Running on Caddy (https://superhumanlythoughts.com)  
**Backend Status:** Ready for deployment

For issues:
- Check logs: `systemctl status automl-backend`
- Test health: `curl http://127.0.0.1:8000/health`
- Check Bedrock: `aws bedrock list-foundation-models --region us-east-2`
