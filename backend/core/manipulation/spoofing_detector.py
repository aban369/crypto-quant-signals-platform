"""
Spoofing and Price Manipulation in Order-Driven Markets
Detection of market manipulation patterns

Spoofing: Placing large orders with intent to cancel before execution
Layering: Multiple orders at different price levels
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
from collections import deque


@dataclass
class ManipulationAlert:
    """Market manipulation alert"""
    timestamp: pd.Timestamp
    manipulation_type: str  # SPOOFING, LAYERING, WASH_TRADING, PUMP_DUMP
    severity: float  # 0-1
    side: str  # BID, ASK, BOTH
    evidence: Dict
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': str(self.timestamp),
            'type': self.manipulation_type,
            'severity': self.severity,
            'side': self.side,
            'evidence': self.evidence,
            'confidence': self.confidence
        }


class SpoofingDetector:
    """
    Detect spoofing patterns in order book
    
    Spoofing characteristics:
    1. Large orders placed far from mid-price
    2. Orders cancelled before execution
    3. Repeated pattern of place-cancel
    4. Asymmetric order book (one-sided pressure)
    """
    
    def __init__(self,
                 size_threshold: float = 10.0,
                 cancel_rate_threshold: float = 0.8,
                 window_size: int = 100):
        """
        Args:
            size_threshold: Minimum order size ratio to average
            cancel_rate_threshold: Minimum cancel rate to flag
            window_size: Number of events to track
        """
        self.size_threshold = size_threshold
        self.cancel_rate_threshold = cancel_rate_threshold
        self.window_size = window_size
        
        self.order_history = deque(maxlen=window_size)
        self.cancel_history = deque(maxlen=window_size)
        
    def detect_spoofing(self,
                       orderbook: Dict,
                       recent_orders: List[Dict],
                       recent_cancels: List[Dict]) -> List[ManipulationAlert]:
        """
        Detect spoofing patterns
        
        Args:
            orderbook: Current order book
            recent_orders: Recent order placements
            recent_cancels: Recent order cancellations
            
        Returns:
            List of manipulation alerts
        """
        alerts = []
        
        # Update history
        self.order_history.extend(recent_orders)
        self.cancel_history.extend(recent_cancels)
        
        # Check for large orders
        large_orders = self._find_large_orders(orderbook)
        
        if not large_orders:
            return alerts
        
        # Check cancel rate
        cancel_rate = self._calculate_cancel_rate()
        
        if cancel_rate > self.cancel_rate_threshold:
            # Potential spoofing
            severity = min(cancel_rate, 1.0)
            
            # Determine side
            bid_pressure = sum([1 for o in large_orders if o['side'] == 'bid'])
            ask_pressure = sum([1 for o in large_orders if o['side'] == 'ask'])
            
            if bid_pressure > ask_pressure * 2:
                side = "BID"
            elif ask_pressure > bid_pressure * 2:
                side = "ASK"
            else:
                side = "BOTH"
            
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="SPOOFING",
                severity=severity,
                side=side,
                evidence={
                    'cancel_rate': cancel_rate,
                    'large_orders': len(large_orders),
                    'avg_order_size': np.mean([o['size'] for o in large_orders])
                },
                confidence=severity
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _find_large_orders(self, orderbook: Dict) -> List[Dict]:
        """Find unusually large orders"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Calculate average size
        all_sizes = []
        all_sizes.extend([float(b[1]) for b in bids[:20]])
        all_sizes.extend([float(a[1]) for a in asks[:20]])
        
        if not all_sizes:
            return []
        
        avg_size = np.mean(all_sizes)
        threshold = avg_size * self.size_threshold
        
        large_orders = []
        
        # Check bids
        for bid in bids[:20]:
            size = float(bid[1])
            if size > threshold:
                large_orders.append({
                    'side': 'bid',
                    'price': float(bid[0]),
                    'size': size
                })
        
        # Check asks
        for ask in asks[:20]:
            size = float(ask[1])
            if size > threshold:
                large_orders.append({
                    'side': 'ask',
                    'price': float(ask[0]),
                    'size': size
                })
        
        return large_orders
    
    def _calculate_cancel_rate(self) -> float:
        """Calculate order cancellation rate"""
        if len(self.order_history) == 0:
            return 0.0
        
        num_cancels = len(self.cancel_history)
        num_orders = len(self.order_history)
        
        cancel_rate = num_cancels / num_orders
        
        return float(cancel_rate)


class LayeringDetector:
    """
    Detect layering manipulation
    
    Layering: Placing multiple orders at different price levels
    to create false impression of supply/demand
    """
    
    def __init__(self, min_layers: int = 5):
        """
        Args:
            min_layers: Minimum number of layers to flag
        """
        self.min_layers = min_layers
        
    def detect_layering(self, orderbook: Dict) -> List[ManipulationAlert]:
        """
        Detect layering patterns
        
        Args:
            orderbook: Order book snapshot
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check bid side
        bid_layers = self._count_layers(orderbook.get('bids', []))
        
        if bid_layers >= self.min_layers:
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="LAYERING",
                severity=min(bid_layers / 10.0, 1.0),
                side="BID",
                evidence={'num_layers': bid_layers},
                confidence=0.7
            )
            alerts.append(alert)
        
        # Check ask side
        ask_layers = self._count_layers(orderbook.get('asks', []))
        
        if ask_layers >= self.min_layers:
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="LAYERING",
                severity=min(ask_layers / 10.0, 1.0),
                side="ASK",
                evidence={'num_layers': ask_layers},
                confidence=0.7
            )
            alerts.append(alert)
        
        return alerts
    
    def _count_layers(self, orders: List) -> int:
        """Count number of similar-sized order layers"""
        if len(orders) < 5:
            return 0
        
        # Get sizes
        sizes = [float(o[1]) for o in orders[:20]]
        
        # Find clusters of similar sizes
        layers = 0
        i = 0
        
        while i < len(sizes) - 1:
            current_size = sizes[i]
            cluster_size = 1
            
            # Count similar sizes
            for j in range(i + 1, len(sizes)):
                if abs(sizes[j] - current_size) / current_size < 0.1:  # Within 10%
                    cluster_size += 1
                else:
                    break
            
            if cluster_size >= 2:
                layers += 1
            
            i += cluster_size
        
        return layers


class WashTradingDetector:
    """
    Detect wash trading (self-trading)
    
    Wash trading: Buying and selling to oneself to create
    false impression of volume
    """
    
    def __init__(self):
        self.trade_history = deque(maxlen=1000)
        
    def detect_wash_trading(self, trades: List[Dict]) -> List[ManipulationAlert]:
        """
        Detect wash trading patterns
        
        Characteristics:
        1. Simultaneous buy and sell at same price
        2. Round-trip trades
        3. No price impact despite volume
        
        Args:
            trades: Recent trades
            
        Returns:
            List of alerts
        """
        alerts = []
        
        self.trade_history.extend(trades)
        
        # Look for simultaneous opposite trades
        suspicious_pairs = self._find_suspicious_pairs()
        
        if len(suspicious_pairs) > 5:
            severity = min(len(suspicious_pairs) / 20.0, 1.0)
            
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="WASH_TRADING",
                severity=severity,
                side="BOTH",
                evidence={
                    'suspicious_pairs': len(suspicious_pairs),
                    'total_volume': sum([p['volume'] for p in suspicious_pairs])
                },
                confidence=0.6
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _find_suspicious_pairs(self) -> List[Dict]:
        """Find suspicious trade pairs"""
        suspicious = []
        
        trades = list(self.trade_history)
        
        for i in range(len(trades) - 1):
            for j in range(i + 1, min(i + 10, len(trades))):
                trade1 = trades[i]
                trade2 = trades[j]
                
                # Check if opposite sides at same price
                if (trade1.get('side') != trade2.get('side') and
                    abs(trade1.get('price', 0) - trade2.get('price', 0)) < 0.01 and
                    abs(trade1.get('volume', 0) - trade2.get('volume', 0)) < 0.01):
                    
                    suspicious.append({
                        'price': trade1.get('price'),
                        'volume': trade1.get('volume'),
                        'time_diff': abs(trade1.get('timestamp', 0) - trade2.get('timestamp', 0))
                    })
        
        return suspicious


class PumpDumpDetector:
    """
    Detect pump and dump schemes
    
    Characteristics:
    1. Rapid price increase with high volume
    2. Followed by rapid price decrease
    3. Coordinated buying/selling
    """
    
    def __init__(self,
                 price_threshold: float = 0.1,
                 volume_threshold: float = 3.0):
        """
        Args:
            price_threshold: Minimum price change (10%)
            volume_threshold: Volume multiplier vs average
        """
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        
    def detect_pump_dump(self,
                        current_price: float,
                        current_volume: float) -> List[ManipulationAlert]:
        """
        Detect pump and dump patterns
        
        Args:
            current_price: Current price
            current_volume: Current volume
            
        Returns:
            List of alerts
        """
        alerts = []
        
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        if len(self.price_history) < 20:
            return alerts
        
        # Calculate price change
        prices = np.array(self.price_history)
        price_change = (prices[-1] - prices[-20]) / prices[-20]
        
        # Calculate volume spike
        volumes = np.array(self.volume_history)
        avg_volume = np.mean(volumes[:-5])
        current_vol_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Detect pump
        if price_change > self.price_threshold and current_vol_ratio > self.volume_threshold:
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="PUMP",
                severity=min(price_change / 0.3, 1.0),
                side="BUY",
                evidence={
                    'price_change': price_change * 100,
                    'volume_ratio': current_vol_ratio
                },
                confidence=0.8
            )
            alerts.append(alert)
        
        # Detect dump
        elif price_change < -self.price_threshold and current_vol_ratio > self.volume_threshold:
            alert = ManipulationAlert(
                timestamp=pd.Timestamp.now(),
                manipulation_type="DUMP",
                severity=min(abs(price_change) / 0.3, 1.0),
                side="SELL",
                evidence={
                    'price_change': price_change * 100,
                    'volume_ratio': current_vol_ratio
                },
                confidence=0.8
            )
            alerts.append(alert)
        
        return alerts


class ManipulationDetectionSystem:
    """
    Comprehensive manipulation detection system
    """
    
    def __init__(self):
        self.spoofing_detector = SpoofingDetector()
        self.layering_detector = LayeringDetector()
        self.wash_trading_detector = WashTradingDetector()
        self.pump_dump_detector = PumpDumpDetector()
        
    def detect_all(self,
                  orderbook: Dict,
                  trades: List[Dict],
                  orders: List[Dict],
                  cancels: List[Dict],
                  current_price: float,
                  current_volume: float) -> List[ManipulationAlert]:
        """
        Run all manipulation detectors
        
        Args:
            orderbook: Order book snapshot
            trades: Recent trades
            orders: Recent orders
            cancels: Recent cancellations
            current_price: Current price
            current_volume: Current volume
            
        Returns:
            List of all alerts
        """
        all_alerts = []
        
        # Spoofing
        all_alerts.extend(
            self.spoofing_detector.detect_spoofing(orderbook, orders, cancels)
        )
        
        # Layering
        all_alerts.extend(
            self.layering_detector.detect_layering(orderbook)
        )
        
        # Wash trading
        all_alerts.extend(
            self.wash_trading_detector.detect_wash_trading(trades)
        )
        
        # Pump & dump
        all_alerts.extend(
            self.pump_dump_detector.detect_pump_dump(current_price, current_volume)
        )
        
        return all_alerts
