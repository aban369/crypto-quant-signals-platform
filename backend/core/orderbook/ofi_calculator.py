"""
Multi-Level Order Flow Imbalance in a Limit Order Book
Implementation of OFI calculation across multiple price levels

OFI predicts short-term price movements based on order book changes
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class OFISignal:
    """Order Flow Imbalance signal"""
    timestamp: pd.Timestamp
    ofi_1: float  # Level 1 OFI
    ofi_5: float  # Level 5 OFI
    ofi_10: float  # Level 10 OFI
    ofi_total: float  # Aggregate OFI
    predicted_direction: str  # UP, DOWN, NEUTRAL
    confidence: float  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'ofi_1': self.ofi_1,
            'ofi_5': self.ofi_5,
            'ofi_10': self.ofi_10,
            'ofi_total': self.ofi_total,
            'direction': self.predicted_direction,
            'confidence': self.confidence
        }


class OrderFlowImbalance:
    """
    Calculate Order Flow Imbalance (OFI) at multiple levels
    
    OFI measures the imbalance between buy and sell pressure
    in the order book, which predicts short-term price movements.
    
    Formula:
    OFI(t) = Σ [ΔBid_volume(i,t) - ΔAsk_volume(i,t)]
    
    where i = price level
    """
    
    def __init__(self, levels: List[int] = [1, 5, 10, 20]):
        """
        Args:
            levels: Price levels to calculate OFI for
        """
        self.levels = levels
        self.prev_orderbook = None
        self.ofi_history = []
        
    def calculate_ofi(self,
                     current_orderbook: Dict,
                     prev_orderbook: Dict = None) -> Dict[int, float]:
        """
        Calculate OFI at multiple levels
        
        Args:
            current_orderbook: Current order book snapshot
            prev_orderbook: Previous order book snapshot
            
        Returns:
            Dictionary mapping level -> OFI value
        """
        if prev_orderbook is None:
            prev_orderbook = self.prev_orderbook
            
        if prev_orderbook is None:
            # First call, no previous data
            self.prev_orderbook = current_orderbook
            return {level: 0.0 for level in self.levels}
        
        ofi_values = {}
        
        for level in self.levels:
            ofi = self._calculate_level_ofi(
                current_orderbook,
                prev_orderbook,
                level
            )
            ofi_values[level] = ofi
        
        # Update previous orderbook
        self.prev_orderbook = current_orderbook
        
        return ofi_values
    
    def _calculate_level_ofi(self,
                            current_ob: Dict,
                            prev_ob: Dict,
                            level: int) -> float:
        """
        Calculate OFI for a specific level
        
        OFI = Σ(i=1 to level) [ΔBid_i - ΔAsk_i]
        
        where:
        ΔBid_i = change in bid volume at level i
        ΔAsk_i = change in ask volume at level i
        """
        current_bids = current_ob.get('bids', [])
        current_asks = current_ob.get('asks', [])
        prev_bids = prev_ob.get('bids', [])
        prev_asks = prev_ob.get('asks', [])
        
        ofi = 0.0
        
        # Calculate for each level up to 'level'
        for i in range(min(level, len(current_bids), len(prev_bids))):
            # Current volumes
            curr_bid_vol = float(current_bids[i][1]) if i < len(current_bids) else 0
            curr_ask_vol = float(current_asks[i][1]) if i < len(current_asks) else 0
            
            # Previous volumes
            prev_bid_vol = float(prev_bids[i][1]) if i < len(prev_bids) else 0
            prev_ask_vol = float(prev_asks[i][1]) if i < len(prev_asks) else 0
            
            # Changes
            delta_bid = curr_bid_vol - prev_bid_vol
            delta_ask = curr_ask_vol - prev_ask_vol
            
            # OFI contribution
            ofi += (delta_bid - delta_ask)
        
        return float(ofi)
    
    def calculate_weighted_ofi(self,
                              current_orderbook: Dict,
                              prev_orderbook: Dict = None) -> float:
        """
        Calculate weighted OFI across all levels
        
        Closer levels have higher weight
        
        Args:
            current_orderbook: Current order book
            prev_orderbook: Previous order book
            
        Returns:
            Weighted OFI value
        """
        ofi_values = self.calculate_ofi(current_orderbook, prev_orderbook)
        
        # Weights: exponentially decreasing
        weights = {
            1: 0.4,
            5: 0.3,
            10: 0.2,
            20: 0.1
        }
        
        weighted_ofi = 0.0
        for level, ofi in ofi_values.items():
            weight = weights.get(level, 1.0 / level)
            weighted_ofi += weight * ofi
        
        return float(weighted_ofi)
    
    def generate_signal(self,
                       current_orderbook: Dict,
                       threshold: float = 100.0) -> OFISignal:
        """
        Generate trading signal based on OFI
        
        Args:
            current_orderbook: Current order book
            threshold: OFI threshold for signal generation
            
        Returns:
            OFISignal object
        """
        ofi_values = self.calculate_ofi(current_orderbook)
        
        ofi_1 = ofi_values.get(1, 0.0)
        ofi_5 = ofi_values.get(5, 0.0)
        ofi_10 = ofi_values.get(10, 0.0)
        
        # Calculate total OFI
        ofi_total = self.calculate_weighted_ofi(current_orderbook)
        
        # Determine direction
        if ofi_total > threshold:
            direction = "UP"
            confidence = min(abs(ofi_total) / (threshold * 3), 1.0)
        elif ofi_total < -threshold:
            direction = "DOWN"
            confidence = min(abs(ofi_total) / (threshold * 3), 1.0)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        signal = OFISignal(
            timestamp=pd.Timestamp.now(),
            ofi_1=ofi_1,
            ofi_5=ofi_5,
            ofi_10=ofi_10,
            ofi_total=ofi_total,
            predicted_direction=direction,
            confidence=confidence
        )
        
        self.ofi_history.append(signal)
        
        return signal
    
    def calculate_ofi_momentum(self, window: int = 10) -> float:
        """
        Calculate OFI momentum (rate of change)
        
        Args:
            window: Lookback window
            
        Returns:
            OFI momentum
        """
        if len(self.ofi_history) < window:
            return 0.0
        
        recent_ofi = [s.ofi_total for s in self.ofi_history[-window:]]
        
        # Linear regression slope
        x = np.arange(len(recent_ofi))
        coeffs = np.polyfit(x, recent_ofi, 1)
        momentum = coeffs[0]
        
        return float(momentum)
    
    def calculate_ofi_volatility(self, window: int = 20) -> float:
        """
        Calculate OFI volatility
        
        Args:
            window: Lookback window
            
        Returns:
            OFI standard deviation
        """
        if len(self.ofi_history) < window:
            return 0.0
        
        recent_ofi = [s.ofi_total for s in self.ofi_history[-window:]]
        volatility = np.std(recent_ofi)
        
        return float(volatility)


class EnhancedOFI:
    """
    Enhanced OFI with additional features
    Based on "Enhancing Trading Strategies with Order Book Signals"
    """
    
    def __init__(self):
        self.ofi_calculator = OrderFlowImbalance()
        
    def calculate_volume_imbalance(self, orderbook: Dict) -> float:
        """
        Calculate volume imbalance at best bid/ask
        
        VI = (Bid_volume - Ask_volume) / (Bid_volume + Ask_volume)
        
        Args:
            orderbook: Order book snapshot
            
        Returns:
            Volume imbalance [-1, 1]
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        bid_vol = float(bids[0][1])
        ask_vol = float(asks[0][1])
        
        total_vol = bid_vol + ask_vol
        
        if total_vol == 0:
            return 0.0
        
        vi = (bid_vol - ask_vol) / total_vol
        
        return float(vi)
    
    def calculate_depth_imbalance(self, 
                                 orderbook: Dict,
                                 levels: int = 10) -> float:
        """
        Calculate depth imbalance across multiple levels
        
        DI = (Σ Bid_depth - Σ Ask_depth) / (Σ Bid_depth + Σ Ask_depth)
        
        Args:
            orderbook: Order book snapshot
            levels: Number of levels to consider
            
        Returns:
            Depth imbalance [-1, 1]
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        bid_depth = sum([float(b[1]) for b in bids[:levels]])
        ask_depth = sum([float(a[1]) for a in asks[:levels]])
        
        total_depth = bid_depth + ask_depth
        
        if total_depth == 0:
            return 0.0
        
        di = (bid_depth - ask_depth) / total_depth
        
        return float(di)
    
    def calculate_spread(self, orderbook: Dict) -> float:
        """
        Calculate bid-ask spread
        
        Args:
            orderbook: Order book snapshot
            
        Returns:
            Spread in basis points
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price * 10000  # bps
        
        return float(spread)
    
    def calculate_microprice(self, orderbook: Dict) -> float:
        """
        Calculate microprice (volume-weighted mid-price)
        
        Microprice = (Bid_price * Ask_volume + Ask_price * Bid_volume) / 
                     (Bid_volume + Ask_volume)
        
        Args:
            orderbook: Order book snapshot
            
        Returns:
            Microprice
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        bid_price = float(bids[0][0])
        bid_vol = float(bids[0][1])
        ask_price = float(asks[0][0])
        ask_vol = float(asks[0][1])
        
        total_vol = bid_vol + ask_vol
        
        if total_vol == 0:
            return (bid_price + ask_price) / 2
        
        microprice = (bid_price * ask_vol + ask_price * bid_vol) / total_vol
        
        return float(microprice)
    
    def calculate_all_features(self, orderbook: Dict) -> Dict:
        """
        Calculate all order book features
        
        Args:
            orderbook: Order book snapshot
            
        Returns:
            Dictionary of features
        """
        features = {
            'volume_imbalance': self.calculate_volume_imbalance(orderbook),
            'depth_imbalance': self.calculate_depth_imbalance(orderbook),
            'spread': self.calculate_spread(orderbook),
            'microprice': self.calculate_microprice(orderbook),
        }
        
        # Add OFI
        ofi_signal = self.ofi_calculator.generate_signal(orderbook)
        features.update(ofi_signal.to_dict())
        
        return features
