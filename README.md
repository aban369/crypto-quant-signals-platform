# 🚀 Crypto Quant Signals Platform

**Advanced Cryptocurrency Analysis & Trading Signals System**  
*Implementing 17+ Research Papers in Econophysics, Market Microstructure, Deep Learning & Reinforcement Learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.0+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

---

## 📚 Research Papers Implemented

### **Econophysics & Statistical Physics**
1. **An Empirical Analysis of Financial Markets: An Econophysics Approach**
   - Statistical physics models for price dynamics
   - Power-law distributions and fat tails
   - Correlation analysis and collective behavior

2. **Thermodynamic Analysis of Financial Markets**
   - Order book temperature calculation
   - Entropy-based market state detection
   - Phase transition identification

3. **An Empirical Analysis on Financial Markets: Insights from Statistical Physics**
   - Agent-based modeling
   - Complexity measures
   - Market efficiency metrics

### **Order Book Dynamics & Microstructure**
4. **Enhancing Trading Strategies with Order Book Signals**
   - Level-2 order book features
   - Bid-ask spread dynamics
   - Volume imbalance indicators

5. **Multi-Level Order Flow Imbalance in a Limit Order Book**
   - Multi-level OFI calculation
   - Price impact prediction
   - Liquidity analysis

6. **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books**
   - CNN architecture for LOB prediction
   - Mid-price movement forecasting
   - Feature extraction from order book snapshots

7. **Econometric Models of Limit Order Executions**
   - Execution probability models
   - Queue position dynamics
   - Fill rate prediction

8. **Optimal Liquidation in a Level I Limit Order Book for Large Tick Stocks**
   - Optimal execution algorithms
   - Market impact minimization
   - TWAP/VWAP strategies

9. **An Introduction to Market Microstructure Theory**
   - Bid-ask spread decomposition
   - Information asymmetry measures
   - Price discovery mechanisms

### **Flash Crashes & Market Anomalies**
10. **What Really Causes Large Price Changes?**
    - Extreme event detection
    - Volatility clustering
    - Jump diffusion models

11. **Classification of Flash Crashes Using the Hawkes (p,q) Framework**
    - Hawkes process implementation
    - Self-exciting point processes
    - Flash crash prediction
    - Cascade detection

12. **Spoofing and Price Manipulation in Order-Driven Markets**
    - Spoofing detection algorithms
    - Layering pattern recognition
    - Manipulation indicators

### **Reinforcement Learning & Portfolio Optimization**
13. **Cryptocurrency Futures Portfolio Trading System Using Reinforcement Learning**
    - Multi-asset RL trading
    - PPO/A3C algorithms
    - Risk-adjusted returns

14. **Tfin Crypto: From Speculation to Optimization in Risk Managed Crypto Portfolio Allocation**
    - Portfolio optimization framework
    - Risk parity strategies
    - Sharpe ratio maximization

15. **FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading**
    - Ensemble RL methods
    - Risk-aware reward functions
    - Position sizing optimization

16. **Machine Learning Approaches to Cryptocurrency Trading Optimization**
    - Comparative ML models (LSTM, GRU, Transformer)
    - Feature engineering
    - Backtesting framework

### **Advanced Topics**
17. **Arbitrage in Perpetual Contracts**
    - Funding rate arbitrage
    - Basis trading strategies
    - Cross-exchange opportunities

18. **A Multifactor Regime-Switching Model for Inter-Trade Durations**
    - Regime detection
    - Markov switching models
    - Volatility forecasting

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │ Signals  │  │Analytics │  │Portfolio │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  REST    │  │WebSocket │  │  Auth    │  │  Cache   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Engine (Python)                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Data Ingestion (Binance, Bybit, OKX WebSockets)  │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Order Book│  │ Hawkes   │  │ DeepLOB  │  │   RL     │   │
│  │ Engine   │  │ Process  │  │   CNN    │  │ Agents   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Thermo-   │  │  OFI     │  │ Flash    │  │Portfolio │   │
│  │dynamics  │  │Calculator│  │ Crash    │  │Optimizer │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Layer (PostgreSQL + Redis)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Time-Series│ │ Order    │  │ Signals  │  │  Models  │   │
│  │   Data    │ │  Books   │  │  Cache   │  │  Store   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features

### **1. Real-Time Order Book Analysis**
- Multi-level order flow imbalance (OFI) calculation
- Bid-ask spread dynamics tracking
- Liquidity heatmaps and depth visualization
- Order book temperature and entropy metrics

### **2. Econophysics Indicators**
- **Temperature**: Measures market "heat" based on order book activity
- **Entropy**: Quantifies market disorder and uncertainty
- **Phase Transitions**: Detects regime changes
- **Power-Law Analysis**: Identifies fat-tail distributions

### **3. Flash Crash Detection**
- Hawkes (p,q) process implementation
- Self-exciting cascade detection
- Real-time anomaly alerts
- Historical flash crash classification

### **4. Deep Learning Predictions**
- **DeepLOB CNN**: Mid-price movement prediction
- **LSTM/GRU**: Time-series forecasting
- **Transformer Models**: Multi-horizon predictions
- **Ensemble Methods**: Combined model outputs

### **5. Reinforcement Learning Trading**
- PPO (Proximal Policy Optimization) agents
- A3C (Asynchronous Advantage Actor-Critic)
- Risk-aware reward functions
- Multi-asset portfolio management

### **6. Market Manipulation Detection**
- Spoofing pattern recognition
- Layering detection
- Wash trading identification
- Pump & dump alerts

### **7. Portfolio Optimization**
- Modern Portfolio Theory (MPT)
- Risk parity allocation
- Kelly Criterion position sizing
- Sharpe ratio maximization

### **8. Arbitrage Opportunities**
- Funding rate arbitrage scanner
- Cross-exchange price discrepancies
- Triangular arbitrage detection
- Statistical arbitrage signals

---

## 🛠️ Technology Stack

### **Backend**
- **Python 3.9+**: Core engine
- **FastAPI**: REST API & WebSockets
- **NumPy/Pandas**: Data processing
- **PyTorch**: Deep learning models
- **Stable-Baselines3**: RL algorithms
- **TA-Lib**: Technical indicators
- **ccxt**: Exchange connectivity

### **Frontend**
- **React 18**: UI framework
- **TypeScript**: Type safety
- **TailwindCSS**: Styling
- **Recharts**: Data visualization
- **Socket.io**: Real-time updates
- **Zustand**: State management

### **Database**
- **PostgreSQL**: Primary database
- **TimescaleDB**: Time-series data
- **Redis**: Caching & pub/sub

### **Infrastructure**
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Nginx**: Reverse proxy
- **Prometheus**: Monitoring
- **Grafana**: Dashboards

---

## 📦 Project Structure

```
crypto-quant-signals-platform/
├── backend/
│   ├── api/                    # FastAPI application
│   │   ├── routes/
│   │   ├── websockets/
│   │   └── middleware/
│   ├── core/                   # Core engine
│   │   ├── econophysics/       # Temperature, entropy, statistical physics
│   │   ├── orderbook/          # Order book processing
│   │   ├── hawkes/             # Hawkes process implementation
│   │   ├── deeplob/            # DeepLOB CNN model
│   │   ├── rl_agents/          # Reinforcement learning
│   │   ├── microstructure/     # Market microstructure models
│   │   ├── flash_crash/        # Flash crash detection
│   │   ├── manipulation/       # Spoofing detection
│   │   └── portfolio/          # Portfolio optimization
│   ├── models/                 # ML/DL model definitions
│   ├── data/                   # Data ingestion & storage
│   ├── utils/                  # Utilities
│   └── tests/                  # Unit tests
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # Page components
│   │   ├── hooks/              # Custom hooks
│   │   ├── services/           # API services
│   │   ├── store/              # State management
│   │   └── utils/              # Utilities
│   └── public/
├── ml_models/                  # Trained models
├── data/                       # Historical data
├── docker/                     # Docker configurations
├── kubernetes/                 # K8s manifests
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
└── tests/                      # Integration tests
```

---

## 🚀 Quick Start

### **Prerequisites**
```bash
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose
```

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/aban369/crypto-quant-signals-platform.git
cd crypto-quant-signals-platform
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Database Setup**
```bash
docker-compose up -d postgres redis
python scripts/init_db.py
```

5. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

6. **Run the Application**
```bash
# Terminal 1: Backend
cd backend
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Data Ingestion
cd backend
python core/data/stream_manager.py
```

### **Docker Deployment**
```bash
docker-compose up -d
```

Access the application at `http://localhost:3000`

---

## 📊 Signal Generation

### **Signal Types**

1. **Econophysics Signals**
   - High temperature + low entropy → Strong trend
   - Low temperature + high entropy → Consolidation
   - Phase transition detected → Regime change

2. **Order Book Signals**
   - OFI > threshold → Buy pressure
   - OFI < -threshold → Sell pressure
   - Spread widening → Liquidity crisis

3. **DeepLOB Predictions**
   - Up movement probability > 0.7 → Long signal
   - Down movement probability > 0.7 → Short signal
   - Stationary probability > 0.6 → No trade

4. **Flash Crash Alerts**
   - Hawkes intensity spike → Potential cascade
   - Branching ratio > 1 → Self-exciting regime
   - Anomaly score > threshold → Flash crash warning

5. **RL Agent Actions**
   - Agent confidence > 0.8 → Execute trade
   - Risk-adjusted Q-value → Position size
   - Portfolio rebalancing signals

---

## 🧪 Backtesting

```python
from core.backtest import BacktestEngine

engine = BacktestEngine(
    strategy="multi_signal_ensemble",
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=100000,
    symbols=["BTC/USDT", "ETH/USDT"]
)

results = engine.run()
print(results.summary())
```

**Metrics Tracked:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

---

## 📈 Performance Optimization

- **Caching**: Redis for hot data
- **Async Processing**: asyncio for I/O operations
- **Batch Processing**: Vectorized NumPy operations
- **Model Optimization**: ONNX runtime for inference
- **Database Indexing**: Optimized queries
- **WebSocket Compression**: Reduced bandwidth

---

## 🔒 Security

- API key encryption
- Rate limiting
- CORS configuration
- Input validation
- SQL injection prevention
- XSS protection

---

## 📖 Documentation

Detailed documentation available in `/docs`:
- [API Reference](docs/api.md)
- [Model Architecture](docs/models.md)
- [Signal Interpretation](docs/signals.md)
- [Deployment Guide](docs/deployment.md)
- [Research Papers Summary](docs/research.md)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with financial advisors before trading.**

---

## 📧 Contact

**Aban Ali**  
Email: raiz.s.group1@gmail.com  
GitHub: [@aban369](https://github.com/aban369)

---

## 🙏 Acknowledgments

Special thanks to the authors of all research papers implemented in this platform. Their groundbreaking work in econophysics, market microstructure, and machine learning has made this project possible.

---

**Built with ❤️ for the crypto quant community**
