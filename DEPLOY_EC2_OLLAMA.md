# Deploy AutoML Backend with Ollama (COMPLETELY FREE)

This guide deploys the AutoML backend with **Ollama** using **Oracle Cloud Free Tier** (always free, no credit card after trial).

## Why This Setup?
- ✅ **ALWAYS FREE** (Oracle Cloud Always-Free tier, no expiration)
- ✅ **Completely free** - no per-token costs
- ✅ **Fast inference** (Mistral 7B optimized)
- ✅ **Better accuracy** than previous Llama models
- ✅ **Self-hosted** (data stays on your server)
- ✅ **Open source** (no vendor lock-in)

## Architecture

```
Oracle Cloud VM (Always Free)
├── Ollama Service (Port 11434) ✅ FREE FOREVER
│   ├── mistral:latest (chat model)
│   └── bge-small-en-v1.5 (embeddings)
├── AutoML Backend (Gunicorn on Port 8000)
│   └── Connects to Ollama via http://localhost:11434/v1
└── Nginx (Port 80/443)
    └── Reverse proxy to backend
```

## Prerequisites

- **Oracle Cloud Account** (free, no credit card needed for always-free tier)
- Always-Free Compute Instance:
  - **VM.Standard.A1.Flex** (Ampere) - Up to 4 OCPU, 24GB RAM ✅ ALWAYS FREE
  - OR **VM.Standard.E2.1.Micro** - 1 OCPU, 1GB RAM (free 12 months)
  - **Mistral 7B needs**: 4GB RAM minimum
  - **bge-small-en-v1.5 needs**: 1GB RAM
- Ubuntu 24.04 LTS

## Step 1: Create Oracle Cloud Always-Free Compute Instance

1. Sign up: https://www.oracle.com/cloud/free/
2. Go to **Compute** → **Instances**
3. Click **Create Instance**
4. Select:
   - **Image**: Ubuntu 24.04
   - **Instance Type**: **VM.Standard.A1.Flex** (Always Free, up to 4 OCPU & 24GB RAM)
   - **Network**: Create new VCN
   - **Subnet**: Create new subnet
5. Add SSH key (download and save the private key)
6. Create Instance

### Security Group Setup (from Oracle Console)
Go to **VCN** → **Security Lists** and allow:
- Port 22 (SSH) - from your IP
- Port 80 (HTTP) - from anywhere
- Port 443 (HTTPS) - from anywhere
- Port 11434 (Ollama) - from localhost only (NOT public)

## Step 2: SSH into Oracle Cloud Instance

```bash
# SSH using the private key downloaded from Oracle
ssh -i /path/to/your-private-key.key ubuntu@<INSTANCE_IP>

# Find instance IP:
# Oracle Console → Instances → Click your instance → Copy Public IP
```

**Important**: Oracle Cloud instances use `ubuntu` user, not `ec2-user`.

## Step 3: Install Ollama

```bash
curl https://ollama.ai/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

## Step 4: Pull Models

```bash
# Start Ollama in background
nohup ollama serve &

# In another terminal, pull models
# This takes 5-10 minutes first time
ollama pull mistral:latest
ollama pull bge-small-en-v1.5

# Verify
ollama list
```

Expected output:
```
NAME                            ID              SIZE      MODIFIED   
mistral:latest                  2ae6254ffb45    4.1 GB    2 hours ago    
bge-small-en-v1.5:latest        ...             ...       ...
```

## Step 5: Set up Ollama Systemd Service

Create `/etc/systemd/system/ollama.service`:

```bash
sudo nano /etc/systemd/system/ollama.service
```

Add:
```ini
[Unit]
Description=Ollama Service
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
User=ollama
Group=ollama
Environment="CUDA_VISIBLE_DEVICES=0"
# For CPU-only, comment out CUDA_VISIBLE_DEVICES

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

Test Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

## Step 6: Deploy AutoML Backend

Clone the repository:
```bash
git clone https://github.com/your-repo/automl-backend.git
cd automl-backend/backend
```

Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file:
```bash
cat > .env << 'EOF'
# Ollama Configuration
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=mistral:latest
OLLAMA_EMBED_MODEL=bge-small-en-v1.5
OLLAMA_OPENAI_API_KEY=ollama

# MongoDB (if using)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/dbname

# CORS Settings
CORS_ORIGINS=https://superhumanlythoughts.com,https://www.superhumanlythoughts.com

# API Auth (optional)
API_AUTH_ENABLED=false
JWT_SECRET_KEY=your-secret-key-change-this

# Email (optional)
EMAIL_ENABLED=false
EOF
```

## Step 7: Set up Backend Systemd Service

Create `/etc/systemd/system/agentic-ai-backend.service`:

```bash
sudo nano /etc/systemd/system/agentic-ai-backend.service
```

Add:
```ini
[Unit]
Description=Gunicorn instance to serve Agentic AI Backend
After=ollama.service
Wants=ollama.service

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/automl-backend/backend
EnvironmentFile=/home/ubuntu/automl-backend/backend/.env
ExecStart=/home/ubuntu/automl-backend/backend/venv/bin/gunicorn \
  -c gunicorn_conf.py \
  -w 4 \
  -b 0.0.0.0:8000 \
  app.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agentic-ai-backend
sudo systemctl start agentic-ai-backend
sudo systemctl status agentic-ai-backend
```

Test backend is running:
```bash
curl http://localhost:8000/health
```

## Step 8: Configure Nginx (Optional but Recommended)

Install Nginx:
```bash
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx
```

Copy Nginx config:
```bash
sudo cp nginx_config_template.conf /etc/nginx/sites-available/automl
sudo ln -s /etc/nginx/sites-available/automl /etc/nginx/sites-enabled/automl
sudo rm /etc/nginx/sites-enabled/default
```

Edit `/etc/nginx/sites-available/automl`:
```nginx
server {
    listen 80;
    server_name superhumanlythoughts.com www.superhumanlythoughts.com;

    location / {
        root /var/www/automl/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Test and enable:
```bash
sudo nginx -t
sudo systemctl restart nginx
```

Enable HTTPS with Let's Encrypt:
```bash
sudo certbot --nginx -d superhumanlythoughts.com -d www.superhumanlythoughts.com
```

## Step 9: Verify Everything is Running

```bash
# Check all services
sudo systemctl status ollama
sudo systemctl status agentic-ai-backend
sudo systemctl status nginx

# Test endpoints
curl http://localhost:11434/api/tags        # Ollama
curl http://localhost:8000/health           # Backend
curl https://superhumanlythoughts.com/api   # Frontend proxy
```

## Monitoring & Logs

```bash
# Ollama logs
sudo journalctl -u ollama -f

# Backend logs
sudo journalctl -u agentic-ai-backend -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## Troubleshooting

### Ollama out of memory
If you see `CUDA out of memory` or `malloc` errors:
```bash
# Reduce Ollama memory usage (CPU mode)
sudo systemctl stop ollama
# Edit service: uncomment CUDA_VISIBLE_DEVICES=0
sudo systemctl start ollama
```

### Backend can't connect to Ollama
```bash
# Check Ollama is listening
sudo netstat -tlnp | grep 11434

# Test connection from backend
curl http://localhost:11434/v1/models
```

### High latency responses
- First inference is slower (model loading)
- Subsequent calls are faster
- If consistently slow: upgrade to t3.2xlarge

## Cost Estimate

| Item | Size | Cost/month |
|------|------|-----------|
| **Oracle Cloud VM.Standard.A1.Flex** | 4 OCPU, 24GB RAM | **$0.00** ✅ ALWAYS FREE |
| **Storage (100GB Block Volume)** | Always Free | **$0.00** ✅ ALWAYS FREE |
| **Data transfer** | Always Free tier | **$0.00** ✅ ALWAYS FREE |
| **Per-token charges** | Ollama local | **$0.00** ✅ NO CHARGES |
| **TOTAL COST** | | **$0.00/month FOREVER** ✅ |

**Key Advantage**: Oracle Cloud Always-Free tier **never expires** (unlike AWS 12-month trial).

## Next Steps

1. Deploy frontend to Oracle Cloud or Vercel (free tier)
2. Monitor performance with logs
3. Keep costs at $0 forever! 🎉

---

## Appendix: Alternative Free Options

### Option A: AWS Free Tier (12 months free)
If you prefer AWS, use **t2.micro** (free for 12 months):
```bash
# Change instance type to t2.micro in the guide above
# BUT: Limited to 1 OCPU, 1GB RAM (not enough for Mistral 7B)
# Best for: Testing only
```

### Option B: Run Locally on Your Machine (Completely Free)
No deployment needed - run Ollama on your laptop:
```bash
# On your local machine:
ollama serve
# Then point frontend to: http://localhost:11434/v1
```
**Best for**: Development only, not production

### Option C: Render.com Free Tier
Free 0.25 CPU machine but unreliable for LLMs. Not recommended.

---
