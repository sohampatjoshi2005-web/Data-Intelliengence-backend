# AutoML Auto-Restart Configuration Guide

## Overview

This document explains the auto-restart setup that prevents the backend and frontend services from stopping unexpectedly on your EC2 production server.

---

## Problem Addressed

Previously, the backend and frontend services would stop automatically after a certain time, and there was no automatic recovery mechanism. This caused:

- ❌ Website going offline
- ❌ API requests failing
- ❌ Manual intervention required to restart services
- ❌ Downtime without alerting

---

## Solution: Auto-Restart with Systemd

The solution uses **systemd** (Linux init system) to:

1. **Automatically restart services** if they crash or stop
2. **Monitor service health** with periodic checks
3. **Limit restart frequency** to prevent restart loops
4. **Log all events** for debugging
5. **Resource limiting** to prevent memory leaks from consuming all resources

---

## Components Deployed

### 1. **automl-backend-improved.service** ✅
- **Location**: `/etc/systemd/system/automl-backend.service`
- **What it does**: 
  - Runs the FastAPI backend (uvicorn)
  - Automatically restarts if it crashes
  - Limits crashes to 5 per 60 seconds (prevents restart loops)
  - Waits 10 seconds before restarting
  - Limits memory to 4GB and CPU to 80%
  - Logs to systemd journal

**Key Configuration:**
```ini
Restart=always                    # Always restart on exit
RestartSec=10                     # Wait 10 seconds before restart
StartLimitInterval=60s            # 60 second window
StartLimitBurst=5                 # Max 5 restarts in that window
MemoryLimit=4G                    # Max 4GB memory
CPUQuota=80%                      # Max 80% CPU usage
```

### 2. **automl-frontend.service** ✅
- **Location**: `/etc/systemd/system/automl-frontend.service`
- **What it does**:
  - Runs Nginx (web server for frontend)
  - Auto-restarts on crash
  - Same restart limits and logging as backend

### 3. **health-check.sh** ✅
- **Location**: `/opt/automl/scripts/health-check.sh`
- **What it does**:
  - Runs periodically (via cron) to check if services are healthy
  - Tests `/health` endpoint on backend
  - Tests `/` endpoint on frontend
  - Auto-restarts services if health check fails
  - Logs results to `/var/log/automl-health-check.log`
  - Sends alerts if services can't be recovered

---

## Deployment Instructions

### Step 1: Deploy Auto-Restart Configuration

Run this from your local machine:

```bash
cd /Users/anuragverma/Downloads/automl-backend
bash deploy-auto-restart.sh
```

**What happens:**
- ✅ Systemd services are installed
- ✅ Health check script is deployed
- ✅ Services are enabled to start on boot
- ✅ All configurations are verified

### Step 2: Verify Installation

SSH into your EC2 instance:

```bash
ssh -i ~/.ssh/id_ed25519 ec2-user@ec2-18-222-158-38.us-east-2.compute.amazonaws.com
```

Check service status:

```bash
# Check backend service
sudo systemctl status automl-backend

# Check frontend service
sudo systemctl status automl-frontend

# List installed services
sudo systemctl list-unit-files | grep automl
```

### Step 3: Start Services

```bash
# Start both services
sudo systemctl start automl-backend
sudo systemctl start automl-frontend

# Verify they're running
curl http://localhost:8000/health
curl http://localhost:80

# Check logs
sudo journalctl -u automl-backend -n 20
sudo journalctl -u automl-frontend -n 20
```

### Step 4: Setup Health Check Cron Job (Optional but Recommended)

Edit crontab:

```bash
crontab -e
```

Add this line to run health check every 5 minutes:

```bash
*/5 * * * * /opt/automl/scripts/health-check.sh
```

---

## How Auto-Restart Works

### Scenario 1: Service Crashes

```
1. Backend service running → gets a bad request → crashes
2. Systemd immediately detects the crash
3. Systemd waits 10 seconds (RestartSec=10)
4. Systemd automatically starts the service again
5. Service is back online within 10 seconds
6. No manual intervention needed ✅
```

### Scenario 2: Memory Leak

```
1. Backend is running
2. Memory usage grows over time due to a bug
3. Memory limit (4GB) is reached
4. Systemd kills the process
5. Systemd automatically restarts it within 10 seconds
6. Service recovers, memory is freed ✅
```

### Scenario 3: Multiple Crashes

```
1. Service crashes repeatedly (5 times in 60 seconds)
2. Systemd detects a restart loop
3. Systemd STOPS trying to restart (prevents infinite loop)
4. Health check script detects service is down
5. Health check logs critical error and attempts manual restart
6. Alert is sent to admin ✅
```

---

## Monitoring & Debugging

### View Recent Logs

```bash
# Backend logs (last 50 lines)
sudo journalctl -u automl-backend -n 50

# Follow logs in real-time
sudo journalctl -u automl-backend -f

# Search for errors
sudo journalctl -u automl-backend | grep -i error
```

### Check Service Status

```bash
# Detailed status
sudo systemctl status automl-backend

# Just running/stopped
sudo systemctl is-active automl-backend

# Show restart count
sudo systemctl show automl-backend -p NRestarts
```

### Manual Restart

```bash
# Restart immediately
sudo systemctl restart automl-backend

# Stop service
sudo systemctl stop automl-backend

# Start service
sudo systemctl start automl-backend

# Reload configuration (without stopping)
sudo systemctl reload automl-backend
```

### View Health Check Results

```bash
# View health check log
tail -f /var/log/automl-health-check.log

# Count restarts from health check
grep "Backend restarted successfully" /var/log/automl-health-check.log | wc -l
```

---

## Configuration Details

### Service Restart Policy

| Setting | Value | Meaning |
|---------|-------|---------|
| `Restart` | `always` | Restart on ANY exit (success or failure) |
| `RestartSec` | `10` | Wait 10 seconds before restarting |
| `StartLimitInterval` | `60s` | Time window for counting restarts |
| `StartLimitBurst` | `5` | Max restarts allowed in `StartLimitInterval` |

### Resource Limits

| Setting | Value | Meaning |
|---------|-------|---------|
| `MemoryLimit` | `4G` | Kill service if it uses >4GB RAM |
| `CPUQuota` | `80%` | Service can use max 80% of 1 CPU core |

### Logging

```bash
# All logs go to systemd journal
sudo journalctl -u automl-backend
sudo journalctl -u automl-frontend

# Also health check logs to file
/var/log/automl-health-check.log
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check for errors
sudo systemctl status automl-backend

# View recent logs
sudo journalctl -u automl-backend -n 20

# Check if port is in use
sudo lsof -i :8000
```

### Failed Health Check

```bash
# Test health endpoint manually
curl -v http://localhost:8000/health

# If it fails, check logs
sudo journalctl -u automl-backend

# Manually restart
sudo systemctl restart automl-backend
```

### Too Many Restarts (Restart Loop)

```bash
# Check restart count
sudo systemctl show automl-backend -p NRestarts

# View when it last crashed
sudo systemctl show automl-backend

# See detailed error in logs
sudo journalctl -u automl-backend --no-pager | tail -50
```

---

## Verification Checklist

After deployment, verify:

- [ ] Both services are installed: `systemctl list-unit-files | grep automl`
- [ ] Services are enabled on boot: `systemctl is-enabled automl-backend`
- [ ] Services start successfully: `systemctl start automl-backend`
- [ ] Health endpoints respond: `curl http://localhost:8000/health`
- [ ] Logs are being generated: `journalctl -u automl-backend -n 5`
- [ ] Health check script runs: `bash /opt/automl/scripts/health-check.sh`
- [ ] Cron job is set (if using): `crontab -l`

---

## What Gets Fixed

With this configuration:

✅ **Service stops?** → Automatically restarts within 10 seconds  
✅ **Memory leak?** → Service is killed and restarted before OOM  
✅ **Port conflict?** → Systemd handles graceful shutdown/restart  
✅ **Crash on startup?** → Systemd prevents infinite restart loop  
✅ **Need to monitor?** → Health check script provides visibility  
✅ **Alerts needed?** → Script can send email on critical failures  

---

## Next Steps

1. **Deploy**: `bash deploy-auto-restart.sh`
2. **Verify**: SSH and check service status
3. **Monitor**: Watch `journalctl -f` during testing
4. **Test**: Intentionally stop a service, watch it restart
5. **Deploy frontend**: Once backend is stable

---

## Support

If issues persist, check:
1. EC2 instance is running (Check AWS console)
2. SSH access works
3. Permissions are correct: `ls -la /etc/systemd/system/automl-*`
4. Services are enabled: `systemctl is-enabled automl-backend`
5. No port conflicts: `sudo lsof -i :8000`

