#!/bin/bash

# Crypto Quant Signals Platform - Quick Start Script
# This script sets up the entire platform with one command

set -e

echo "🚀 Crypto Quant Signals Platform - Quick Start"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env file with your API keys before running in production${NC}"
    echo ""
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p ml_models
mkdir -p data
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Pull Docker images
echo -e "${YELLOW}🐳 Pulling Docker images...${NC}"
docker-compose pull
echo -e "${GREEN}✅ Docker images pulled${NC}"
echo ""

# Build services
echo -e "${YELLOW}🔨 Building services...${NC}"
docker-compose build
echo -e "${GREEN}✅ Services built${NC}"
echo ""

# Start services
echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose up -d
echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "${YELLOW}🏥 Checking service health...${NC}"

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres &> /dev/null; then
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
else
    echo -e "${RED}❌ PostgreSQL is not ready${NC}"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✅ Redis is ready${NC}"
else
    echo -e "${RED}❌ Redis is not ready${NC}"
fi

# Check Backend
if curl -s http://localhost:8000/api/health &> /dev/null; then
    echo -e "${GREEN}✅ Backend API is ready${NC}"
else
    echo -e "${YELLOW}⚠️  Backend API is starting... (may take a few more seconds)${NC}"
fi

# Check Frontend
if curl -s http://localhost:3000 &> /dev/null; then
    echo -e "${GREEN}✅ Frontend is ready${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend is starting... (may take a few more seconds)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "=============================================="
echo "📊 Access the platform:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "🔧 Useful commands:"
echo "   View logs:        docker-compose logs -f"
echo "   Stop services:    docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   View status:      docker-compose ps"
echo ""
echo "📚 Documentation:"
echo "   README:       ./README.md"
echo "   API Docs:     ./docs/API.md"
echo "   Deployment:   ./docs/DEPLOYMENT.md"
echo "   Research:     ./docs/RESEARCH_PAPERS.md"
echo ""
echo "⚠️  Important:"
echo "   - Edit .env file with your API keys for production"
echo "   - Default credentials are for development only"
echo "   - See docs/DEPLOYMENT.md for production setup"
echo ""
echo "=============================================="
echo ""
echo -e "${GREEN}Happy trading! 🚀${NC}"
