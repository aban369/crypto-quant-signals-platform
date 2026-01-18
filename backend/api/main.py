"""
FastAPI Main Application
REST API and WebSocket server for crypto signals platform
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict
import asyncio
import json
from datetime import datetime

# Import core modules
import sys
sys.path.append('..')
from core.signals.signal_generator import SignalGenerator
from core.econophysics.temperature import OrderBookThermodynamics
from core.hawkes.flash_crash_detector import FlashCrashDetector
from core.orderbook.ofi_calculator import OrderFlowImbalance
from core.portfolio.optimizer import PortfolioOptimizer

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Quant Signals Platform",
    description="Advanced cryptocurrency analysis and trading signals",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
signal_generator = None
active_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
websocket_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global signal_generator
    signal_generator = SignalGenerator(symbols=active_symbols)
    print("✅ Signal Generator initialized")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Crypto Quant Signals Platform",
        "version": "1.0.0",
        "status": "running",
        "research_papers": 17,
        "active_symbols": active_symbols
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "signal_generator": signal_generator is not None,
            "websocket_connections": len(websocket_connections)
        }
    }


@app.get("/api/symbols")
async def get_symbols():
    """Get list of supported symbols"""
    return {
        "symbols": active_symbols,
        "count": len(active_symbols)
    }


@app.get("/api/signals/{symbol}")
async def get_signal(symbol: str):
    """
    Get latest signal for a symbol
    
    Args:
        symbol: Trading symbol (e.g., BTC/USDT)
    """
    if symbol not in active_symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    # Mock data for demonstration
    # In production, this would fetch real-time data
    import numpy as np
    
    orderbook = {
        'bids': [[50000 - i*10, 1.5 + i*0.1] for i in range(50)],
        'asks': [[50000 + i*10, 1.5 + i*0.1] for i in range(50)]
    }
    
    trades = [
        {'price': 50000, 'volume': 1.5, 'timestamp': datetime.now().timestamp()}
        for _ in range(100)
    ]
    
    orderbook_history = [orderbook for _ in range(100)]
    
    prices = np.random.randn(1000).cumsum() + 50000
    volumes = np.random.rand(1000) * 10
    timestamps = np.arange(1000) * 1000
    
    # Generate signal
    signal = signal_generator.generate_signal(
        symbol=symbol,
        orderbook=orderbook,
        trades=trades,
        orderbook_history=orderbook_history,
        prices=prices,
        volumes=volumes,
        timestamps=timestamps
    )
    
    return JSONResponse(content=signal.to_dict())


@app.get("/api/signals")
async def get_all_signals():
    """Get signals for all symbols"""
    signals = {}
    
    for symbol in active_symbols:
        try:
            # Mock data
            import numpy as np
            
            orderbook = {
                'bids': [[50000 - i*10, 1.5 + i*0.1] for i in range(50)],
                'asks': [[50000 + i*10, 1.5 + i*0.1] for i in range(50)]
            }
            
            trades = [
                {'price': 50000, 'volume': 1.5, 'timestamp': datetime.now().timestamp()}
                for _ in range(100)
            ]
            
            orderbook_history = [orderbook for _ in range(100)]
            prices = np.random.randn(1000).cumsum() + 50000
            volumes = np.random.rand(1000) * 10
            timestamps = np.arange(1000) * 1000
            
            signal = signal_generator.generate_signal(
                symbol=symbol,
                orderbook=orderbook,
                trades=trades,
                orderbook_history=orderbook_history,
                prices=prices,
                volumes=volumes,
                timestamps=timestamps
            )
            
            signals[symbol] = signal.to_dict()
        except Exception as e:
            signals[symbol] = {"error": str(e)}
    
    return JSONResponse(content=signals)


@app.get("/api/econophysics/{symbol}")
async def get_econophysics(symbol: str):
    """Get econophysics metrics for a symbol"""
    if symbol not in active_symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    # Mock data
    orderbook = {
        'bids': [[50000 - i*10, 1.5 + i*0.1] for i in range(50)],
        'asks': [[50000 + i*10, 1.5 + i*0.1] for i in range(50)]
    }
    
    trades = [
        {'price': 50000, 'volume': 1.5, 'timestamp': datetime.now().timestamp()}
        for _ in range(100)
    ]
    
    thermo = OrderBookThermodynamics()
    state = thermo.get_thermodynamic_state(orderbook, trades)
    
    return {
        "symbol": symbol,
        "temperature": state.temperature,
        "entropy": state.entropy,
        "pressure": state.pressure,
        "volume": state.volume,
        "free_energy": state.free_energy,
        "phase": state.phase(),
        "timestamp": state.timestamp.isoformat()
    }


@app.get("/api/orderbook/{symbol}")
async def get_orderbook_analysis(symbol: str):
    """Get order book analysis"""
    if symbol not in active_symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    # Mock data
    orderbook = {
        'bids': [[50000 - i*10, 1.5 + i*0.1] for i in range(50)],
        'asks': [[50000 + i*10, 1.5 + i*0.1] for i in range(50)]
    }
    
    ofi = OrderFlowImbalance()
    signal = ofi.generate_signal(orderbook)
    
    return {
        "symbol": symbol,
        "ofi_1": signal.ofi_1,
        "ofi_5": signal.ofi_5,
        "ofi_10": signal.ofi_10,
        "ofi_total": signal.ofi_total,
        "direction": signal.predicted_direction,
        "confidence": signal.confidence,
        "timestamp": signal.timestamp.isoformat()
    }


@app.get("/api/flash-crash/{symbol}")
async def get_flash_crash_analysis(symbol: str):
    """Get flash crash analysis"""
    if symbol not in active_symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    # Mock data
    import numpy as np
    prices = np.random.randn(1000).cumsum() + 50000
    volumes = np.random.rand(1000) * 10
    timestamps = np.arange(1000) * 1000
    
    detector = FlashCrashDetector()
    crashes = detector.detect_flash_crash(prices, timestamps, volumes)
    
    crash_prob = detector.predict_crash_probability(float(timestamps[-1]) / 1e9)
    
    return {
        "symbol": symbol,
        "crash_probability": crash_prob,
        "num_crashes_detected": len(crashes),
        "recent_crashes": [
            {
                "timestamp": str(c.timestamp),
                "severity": c.severity,
                "price_drop": c.price_drop,
                "classification": c.classification
            }
            for c in crashes[-5:]
        ]
    }


@app.get("/api/portfolio/optimize")
async def optimize_portfolio(method: str = "sharpe"):
    """
    Optimize portfolio allocation
    
    Args:
        method: Optimization method (sharpe, min_variance, risk_parity)
    """
    import pandas as pd
    import numpy as np
    
    # Mock price data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    prices_data = {}
    
    for symbol in active_symbols:
        prices_data[symbol] = np.random.randn(len(dates)).cumsum() + 50000
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    optimizer = PortfolioOptimizer()
    expected_returns, cov_matrix = optimizer.calculate_returns_covariance(prices_df)
    
    if method == "sharpe":
        allocation = optimizer.optimize_sharpe(expected_returns, cov_matrix, active_symbols)
    elif method == "min_variance":
        allocation = optimizer.optimize_min_variance(cov_matrix, active_symbols)
    elif method == "risk_parity":
        allocation = optimizer.risk_parity(cov_matrix, active_symbols)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    
    return allocation.to_dict()


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signals
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send signals every 5 seconds
            await asyncio.sleep(5)
            
            # Generate signals for all symbols
            signals = {}
            for symbol in active_symbols:
                try:
                    import numpy as np
                    
                    orderbook = {
                        'bids': [[50000 - i*10, 1.5 + i*0.1] for i in range(50)],
                        'asks': [[50000 + i*10, 1.5 + i*0.1] for i in range(50)]
                    }
                    
                    trades = [
                        {'price': 50000, 'volume': 1.5, 'timestamp': datetime.now().timestamp()}
                        for _ in range(100)
                    ]
                    
                    orderbook_history = [orderbook for _ in range(100)]
                    prices = np.random.randn(1000).cumsum() + 50000
                    volumes = np.random.rand(1000) * 10
                    timestamps = np.arange(1000) * 1000
                    
                    signal = signal_generator.generate_signal(
                        symbol=symbol,
                        orderbook=orderbook,
                        trades=trades,
                        orderbook_history=orderbook_history,
                        prices=prices,
                        volumes=volumes,
                        timestamps=timestamps
                    )
                    
                    signals[symbol] = signal.to_dict()
                except Exception as e:
                    signals[symbol] = {"error": str(e)}
            
            # Send to client
            await websocket.send_json({
                "type": "signals_update",
                "data": signals,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        print("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
