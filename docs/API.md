# API Documentation

Complete REST API and WebSocket documentation for the Crypto Quant Signals Platform.

**Base URL**: `http://localhost:8000` (development)  
**Production**: `https://api.yourdomain.com`

---

## Table of Contents

1. [Authentication](#authentication)
2. [REST API Endpoints](#rest-api-endpoints)
3. [WebSocket API](#websocket-api)
4. [Response Formats](#response-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## Authentication

Currently, the API is open for development. Production deployment should implement:

```http
Authorization: Bearer <your_jwt_token>
```

---

## REST API Endpoints

### Health & Status

#### GET /
Get API information

**Response**:
```json
{
  "name": "Crypto Quant Signals Platform",
  "version": "1.0.0",
  "status": "running",
  "research_papers": 17,
  "active_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
}
```

#### GET /api/health
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T13:00:00Z",
  "components": {
    "signal_generator": true,
    "websocket_connections": 5
  }
}
```

---

### Symbols

#### GET /api/symbols
Get list of supported trading symbols

**Response**:
```json
{
  "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  "count": 3
}
```

---

### Trading Signals

#### GET /api/signals/{symbol}
Get latest comprehensive trading signal for a symbol

**Parameters**:
- `symbol` (path): Trading symbol (e.g., BTC/USDT)

**Example**:
```http
GET /api/signals/BTC/USDT
```

**Response**:
```json
{
  "timestamp": "2026-01-18T13:00:00Z",
  "symbol": "BTC/USDT",
  "direction": "LONG",
  "strength": 4,
  "confidence": 0.78,
  "econophysics": {
    "temperature": 1.45,
    "entropy": 0.62,
    "pressure": 0.35,
    "phase": "TRENDING_HOT_TRENDING",
    "power_law_exponent": 2.8,
    "hurst_exponent": 0.65,
    "regime": "TRENDING_HOT_TRENDING"
  },
  "orderbook": {
    "ofi_total": 245.6,
    "ofi_direction": "UP",
    "ofi_confidence": 0.82,
    "volume_imbalance": 0.28,
    "depth_imbalance": 0.31,
    "spread": 12.5,
    "microprice": 50125.34
  },
  "deeplob": {
    "direction": "UP",
    "up_prob": 0.75,
    "down_prob": 0.15,
    "stationary_prob": 0.10,
    "confidence": 0.75
  },
  "hawkes": {
    "crash_probability": 0.12,
    "num_crashes_detected": 0,
    "recent_crashes": []
  },
  "rl": {
    "action": "BUY",
    "position_size": 0.25,
    "confidence": 0.72,
    "expected_return": 0.035
  },
  "manipulation": null,
  "expected_return": 0.042,
  "risk_score": 0.28,
  "sharpe_estimate": 1.85,
  "entry_price": 50000.00,
  "stop_loss": 48500.00,
  "take_profit": 53000.00,
  "position_size": 0.195,
  "reasoning": "LONG STRONG: Market phase: TRENDING_HOT_TRENDING | Strong order flow imbalance: UP | DeepLOB predicts UP with 75.0% confidence | RL agent recommends BUY",
  "warnings": []
}
```

#### GET /api/signals
Get signals for all symbols

**Response**:
```json
{
  "BTC/USDT": { /* signal object */ },
  "ETH/USDT": { /* signal object */ },
  "SOL/USDT": { /* signal object */ }
}
```

---

### Econophysics Analysis

#### GET /api/econophysics/{symbol}
Get thermodynamic and statistical physics metrics

**Parameters**:
- `symbol` (path): Trading symbol

**Example**:
```http
GET /api/econophysics/BTC/USDT
```

**Response**:
```json
{
  "symbol": "BTC/USDT",
  "temperature": 1.45,
  "entropy": 0.62,
  "pressure": 0.35,
  "volume": 12500.5,
  "free_energy": 0.55,
  "phase": "TRENDING_HOT",
  "timestamp": "2026-01-18T13:00:00Z"
}
```

**Interpretation**:
- **Temperature** (0-3+): Market activity level
  - < 0.5: Cold (low volatility)
  - 0.5-1.5: Normal
  - > 1.5: Hot (high volatility)
  
- **Entropy** (0-1): Market disorder
  - < 0.3: Ordered (concentrated liquidity)
  - 0.3-0.7: Normal
  - > 0.7: Chaotic (dispersed liquidity)
  
- **Pressure** (-1 to 1): Buy/sell pressure
  - < -0.3: Strong sell pressure
  - -0.3 to 0.3: Balanced
  - > 0.3: Strong buy pressure

---

### Order Book Analysis

#### GET /api/orderbook/{symbol}
Get order flow imbalance and microstructure metrics

**Parameters**:
- `symbol` (path): Trading symbol

**Example**:
```http
GET /api/orderbook/BTC/USDT
```

**Response**:
```json
{
  "symbol": "BTC/USDT",
  "ofi_1": 45.2,
  "ofi_5": 123.8,
  "ofi_10": 245.6,
  "ofi_total": 245.6,
  "direction": "UP",
  "confidence": 0.82,
  "timestamp": "2026-01-18T13:00:00Z"
}
```

**Interpretation**:
- **OFI** (Order Flow Imbalance):
  - Positive: Buy pressure
  - Negative: Sell pressure
  - Magnitude indicates strength
  
- **Direction**:
  - UP: Bullish
  - DOWN: Bearish
  - NEUTRAL: No clear direction

---

### Flash Crash Detection

#### GET /api/flash-crash/{symbol}
Get flash crash analysis using Hawkes process

**Parameters**:
- `symbol` (path): Trading symbol

**Example**:
```http
GET /api/flash-crash/BTC/USDT
```

**Response**:
```json
{
  "symbol": "BTC/USDT",
  "crash_probability": 0.12,
  "num_crashes_detected": 2,
  "recent_crashes": [
    {
      "timestamp": "2026-01-18T10:30:00Z",
      "severity": 0.65,
      "price_drop": 8.5,
      "classification": "MODERATE"
    },
    {
      "timestamp": "2026-01-18T09:15:00Z",
      "severity": 0.35,
      "price_drop": 3.2,
      "classification": "MINOR"
    }
  ]
}
```

**Classifications**:
- **MINOR**: Severity < 0.25
- **MODERATE**: Severity 0.25-0.50
- **SEVERE**: Severity 0.50-0.75
- **EXTREME**: Severity > 0.75

---

### Portfolio Optimization

#### GET /api/portfolio/optimize
Optimize portfolio allocation

**Query Parameters**:
- `method` (optional): Optimization method
  - `sharpe` (default): Maximum Sharpe ratio
  - `min_variance`: Minimum variance
  - `risk_parity`: Equal risk contribution

**Example**:
```http
GET /api/portfolio/optimize?method=sharpe
```

**Response**:
```json
{
  "weights": {
    "BTC/USDT": 0.45,
    "ETH/USDT": 0.35,
    "SOL/USDT": 0.20
  },
  "expected_return": 0.285,
  "volatility": 0.145,
  "sharpe_ratio": 1.83,
  "max_drawdown": 0.0,
  "var_95": 0.0,
  "cvar_95": 0.0
}
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals');

ws.onopen = () => {
  console.log('Connected to signals stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received signals:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from signals stream');
};
```

### Message Format

**Incoming Messages**:
```json
{
  "type": "signals_update",
  "data": {
    "BTC/USDT": { /* signal object */ },
    "ETH/USDT": { /* signal object */ },
    "SOL/USDT": { /* signal object */ }
  },
  "timestamp": "2026-01-18T13:00:00Z"
}
```

**Update Frequency**: Every 5 seconds

---

## Response Formats

### Success Response

```json
{
  "status": "success",
  "data": { /* response data */ },
  "timestamp": "2026-01-18T13:00:00Z"
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "SYMBOL_NOT_FOUND",
    "message": "Symbol BTC/USD not found",
    "details": "Available symbols: BTC/USDT, ETH/USDT, SOL/USDT"
  },
  "timestamp": "2026-01-18T13:00:00Z"
}
```

---

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Codes

| Code | Description |
|------|-------------|
| `SYMBOL_NOT_FOUND` | Trading symbol not supported |
| `INVALID_PARAMETER` | Invalid query parameter |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Internal server error |
| `SERVICE_UNAVAILABLE` | Service temporarily down |

---

## Rate Limiting

**Limits**:
- REST API: 100 requests per minute per IP
- WebSocket: 10 connections per IP

**Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642512000
```

**Rate Limit Exceeded Response**:
```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}
```

---

## Code Examples

### Python

```python
import requests

# Get signal
response = requests.get('http://localhost:8000/api/signals/BTC/USDT')
signal = response.json()

print(f"Direction: {signal['direction']}")
print(f"Confidence: {signal['confidence']}")
print(f"Entry: ${signal['entry_price']}")
print(f"Stop Loss: ${signal['stop_loss']}")
print(f"Take Profit: ${signal['take_profit']}")
```

### JavaScript

```javascript
// Fetch signal
fetch('http://localhost:8000/api/signals/BTC/USDT')
  .then(response => response.json())
  .then(signal => {
    console.log(`Direction: ${signal.direction}`);
    console.log(`Confidence: ${signal.confidence}`);
  });

// WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/signals');
ws.onmessage = (event) => {
  const { data } = JSON.parse(event.data);
  Object.entries(data).forEach(([symbol, signal]) => {
    console.log(`${symbol}: ${signal.direction} (${signal.confidence})`);
  });
};
```

### cURL

```bash
# Get signal
curl http://localhost:8000/api/signals/BTC/USDT

# Get all signals
curl http://localhost:8000/api/signals

# Optimize portfolio
curl "http://localhost:8000/api/portfolio/optimize?method=sharpe"
```

---

## Pagination

For endpoints returning large datasets:

**Query Parameters**:
- `page` (default: 1): Page number
- `per_page` (default: 50, max: 100): Items per page

**Response**:
```json
{
  "data": [ /* items */ ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 250,
    "pages": 5
  }
}
```

---

## Filtering & Sorting

**Query Parameters**:
- `filter[field]`: Filter by field value
- `sort`: Sort field (prefix with `-` for descending)

**Example**:
```http
GET /api/signals?filter[direction]=LONG&sort=-confidence
```

---

## Webhooks (Coming Soon)

Subscribe to signal updates via webhooks:

```http
POST /api/webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["signal_update", "flash_crash_alert"],
  "symbols": ["BTC/USDT", "ETH/USDT"]
}
```

---

## API Versioning

Current version: `v1`

Future versions will be accessible via:
```http
GET /api/v2/signals/BTC/USDT
```

---

## Support

- API Issues: https://github.com/aban369/crypto-quant-signals-platform/issues
- Email: raiz.s.group1@gmail.com
- Documentation: https://docs.bhindi.io

---

**Last Updated**: January 2026  
**API Version**: 1.0.0
