# Deployment Guide

Complete guide for deploying the Crypto Quant Signals Platform in production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- OS: Ubuntu 20.04+ / macOS / Windows with WSL2

**Recommended**:
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 100+ GB NVMe SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for DeepLOB training)

### Software Requirements

```bash
# Required
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose

# Optional
- CUDA 11.8+ (for GPU acceleration)
- Kubernetes (for production)
```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/aban369/crypto-quant-signals-platform.git
cd crypto-quant-signals-platform
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../.env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Run backend
uvicorn api.main:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set up environment
cp .env.example .env.local
# Edit .env.local with your configuration

# Run frontend
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Database Setup

```bash
# Start PostgreSQL
docker run -d \
  --name crypto-quant-postgres \
  -e POSTGRES_DB=crypto_quant \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg14

# Start Redis
docker run -d \
  --name crypto-quant-redis \
  -p 6379:6379 \
  redis:7-alpine
```

---

## Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### Custom Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  backend:
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
      - REDIS_URL=redis://redis:6379
      # Add your API keys
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
```

---

## Production Deployment

### Option 1: Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml crypto-quant

# Scale services
docker service scale crypto-quant_backend=3

# View services
docker service ls
```

### Option 2: Kubernetes

```bash
# Apply configurations
kubectl apply -f kubernetes/

# Check deployments
kubectl get deployments
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment backend --replicas=3
```

### Option 3: Cloud Platforms

#### AWS Deployment

```bash
# Using ECS
aws ecs create-cluster --cluster-name crypto-quant

# Deploy task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Create service
aws ecs create-service \
  --cluster crypto-quant \
  --service-name crypto-quant-api \
  --task-definition crypto-quant-backend \
  --desired-count 2
```

#### Google Cloud Platform

```bash
# Build and push images
gcloud builds submit --tag gcr.io/PROJECT_ID/crypto-quant-backend backend/
gcloud builds submit --tag gcr.io/PROJECT_ID/crypto-quant-frontend frontend/

# Deploy to Cloud Run
gcloud run deploy crypto-quant-backend \
  --image gcr.io/PROJECT_ID/crypto-quant-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### DigitalOcean

```bash
# Create app
doctl apps create --spec .do/app.yaml

# Update app
doctl apps update APP_ID --spec .do/app.yaml
```

---

## Environment Variables

### Backend (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/crypto_quant
REDIS_URL=redis://host:6379

# API Keys
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret

# Security
SECRET_KEY=generate_random_secret_key
JWT_SECRET=generate_random_jwt_secret
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Application
DEBUG=False
LOG_LEVEL=INFO
WORKERS=4

# Models
DEEPLOB_MODEL_PATH=/app/ml_models/deeplob_model.pth
PPO_MODEL_PATH=/app/ml_models/ppo_trader.zip

# Trading
MAX_POSITION_SIZE=0.3
TRANSACTION_COST=0.001
RISK_FREE_RATE=0.02
```

### Frontend (.env.production)

```bash
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com
VITE_ENVIRONMENT=production
```

---

## SSL/TLS Configuration

### Using Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Backend API
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Frontend
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
    }
}
```

---

## Database Migration

### Initial Setup

```bash
cd backend

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

### Backup & Restore

```bash
# Backup
pg_dump -h localhost -U postgres crypto_quant > backup.sql

# Restore
psql -h localhost -U postgres crypto_quant < backup.sql
```

---

## Monitoring & Maintenance

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'crypto-quant-backend'
    static_configs:
      - targets: ['backend:8000']
```

### Grafana Dashboards

Import pre-built dashboards:
- System metrics
- API performance
- Trading signals
- Model accuracy

### Logging

```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/health

# Database health
pg_isready -h localhost -p 5432

# Redis health
redis-cli ping
```

---

## Performance Optimization

### Backend

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)

# Enable caching
from aiocache import Cache

cache = Cache(Cache.REDIS, endpoint="redis", port=6379)
```

### Frontend

```javascript
// Code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));

// Memoization
const MemoizedComponent = React.memo(Component);

// Virtual scrolling for large lists
import { FixedSizeList } from 'react-window';
```

### Database

```sql
-- Create indexes
CREATE INDEX idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX idx_signals_symbol ON signals(symbol);

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;
SELECT create_hypertable('signals', 'timestamp');
```

---

## Scaling

### Horizontal Scaling

```bash
# Scale backend
docker-compose up -d --scale backend=3

# Load balancer configuration
upstream backend {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

### Vertical Scaling

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

---

## Troubleshooting

### Common Issues

**Backend won't start**:
```bash
# Check logs
docker-compose logs backend

# Verify database connection
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://...'); engine.connect()"
```

**Frontend build fails**:
```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+
```

**Database connection errors**:
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Test connection
psql -h localhost -U postgres -d crypto_quant
```

---

## Security Checklist

- [ ] Change default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Set up API rate limiting
- [ ] Enable CORS properly
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Enable database encryption
- [ ] Implement authentication
- [ ] Set up monitoring alerts

---

## Backup Strategy

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Database backup
pg_dump -h localhost -U postgres crypto_quant | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Redis backup
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Model files
tar -czf $BACKUP_DIR/models_$DATE.tar.gz ml_models/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

Add to crontab:
```bash
0 2 * * * /path/to/backup.sh
```

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/aban369/crypto-quant-signals-platform/issues
- Email: raiz.s.group1@gmail.com
- Documentation: https://docs.bhindi.io

---

**Last Updated**: January 2026
