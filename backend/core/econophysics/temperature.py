"""
Thermodynamic Analysis of Financial Markets
Implementation of "Thermodynamic Analysis of Financial Markets: 
Measuring Order Book Dynamics with Temperature and Entropy"

Temperature measures the "heat" or activity level in the order book.
High temperature indicates high volatility and trading activity.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class ThermodynamicState:
    """Represents the thermodynamic state of the market"""
    temperature: float
    entropy: float
    free_energy: float
    pressure: float
    volume: float
    timestamp: pd.Timestamp
    
    def phase(self) -> str:
        """Determine market phase based on thermodynamic properties"""
        if self.temperature > 1.5 and self.entropy < 0.5:
            return "TRENDING_HOT"
        elif self.temperature < 0.5 and self.entropy > 1.5:
            return "CONSOLIDATION_COLD"
        elif self.temperature > 1.0 and self.entropy > 1.0:
            return "VOLATILE_CHAOTIC"
        else:
            return "EQUILIBRIUM"


class OrderBookThermodynamics:
    """
    Calculate thermodynamic properties of order book
    Based on statistical physics approach to financial markets
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = []
        self.volume_history = []
        
    def calculate_temperature(self, 
                            order_book: Dict,
                            trades: List[Dict]) -> float:
        """
        Calculate market temperature using kinetic energy analogy
        
        Temperature = (1/N) * Σ(ΔP_i)² * V_i
        where:
        - ΔP_i = price change
        - V_i = volume
        - N = number of observations
        
        Args:
            order_book: Current order book snapshot
            trades: Recent trades
            
        Returns:
            Temperature value (normalized)
        """
        if not trades:
            return 0.0
            
        # Calculate price volatility (kinetic energy)
        prices = np.array([t['price'] for t in trades[-self.window_size:]])
        volumes = np.array([t['volume'] for t in trades[-self.window_size:]])
        
        if len(prices) < 2:
            return 0.0
            
        # Price changes (velocity)
        price_changes = np.diff(prices)
        
        # Kinetic energy: (1/2) * m * v²
        # Here: volume acts as mass, price change as velocity
        kinetic_energy = 0.5 * volumes[1:] * (price_changes ** 2)
        
        # Temperature is average kinetic energy
        temperature = np.mean(kinetic_energy)
        
        # Normalize by current price
        current_price = prices[-1]
        normalized_temp = temperature / (current_price ** 2) * 1000
        
        return float(normalized_temp)
    
    def calculate_entropy(self, order_book: Dict) -> float:
        """
        Calculate Shannon entropy of order book distribution
        
        Entropy = -Σ p_i * log(p_i)
        where p_i is the probability of price level i
        
        High entropy = high disorder/uncertainty
        Low entropy = concentrated liquidity
        
        Args:
            order_book: Order book with bids and asks
            
        Returns:
            Entropy value
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        # Combine all volumes
        all_volumes = []
        all_volumes.extend([float(b[1]) for b in bids[:50]])  # Top 50 levels
        all_volumes.extend([float(a[1]) for a in asks[:50]])
        
        volumes = np.array(all_volumes)
        
        if volumes.sum() == 0:
            return 0.0
        
        # Calculate probability distribution
        probabilities = volumes / volumes.sum()
        
        # Remove zeros to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def calculate_pressure(self, order_book: Dict) -> float:
        """
        Calculate market pressure (buy vs sell pressure)
        
        Pressure = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
        
        Positive = buy pressure
        Negative = sell pressure
        
        Args:
            order_book: Order book snapshot
            
        Returns:
            Pressure value [-1, 1]
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        # Calculate total volume at each side (top 20 levels)
        bid_volume = sum([float(b[1]) for b in bids[:20]])
        ask_volume = sum([float(a[1]) for a in asks[:20]])
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        pressure = (bid_volume - ask_volume) / total_volume
        
        return float(pressure)
    
    def calculate_free_energy(self, 
                             temperature: float, 
                             entropy: float) -> float:
        """
        Calculate Helmholtz free energy
        
        F = E - T*S
        where:
        - E = internal energy (approximated by price level)
        - T = temperature
        - S = entropy
        
        Free energy indicates market stability
        
        Args:
            temperature: Market temperature
            entropy: Market entropy
            
        Returns:
            Free energy value
        """
        # Internal energy approximated by normalized price variance
        internal_energy = 1.0  # Baseline
        
        free_energy = internal_energy - (temperature * entropy)
        
        return float(free_energy)
    
    def calculate_volume(self, order_book: Dict) -> float:
        """
        Calculate market volume (total liquidity)
        
        Volume = Total available liquidity in order book
        
        Args:
            order_book: Order book snapshot
            
        Returns:
            Total volume
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        total_volume = 0.0
        
        if bids:
            total_volume += sum([float(b[1]) for b in bids[:50]])
        if asks:
            total_volume += sum([float(a[1]) for a in asks[:50]])
        
        return float(total_volume)
    
    def get_thermodynamic_state(self,
                               order_book: Dict,
                               trades: List[Dict]) -> ThermodynamicState:
        """
        Calculate complete thermodynamic state
        
        Args:
            order_book: Current order book
            trades: Recent trades
            
        Returns:
            ThermodynamicState object
        """
        temperature = self.calculate_temperature(order_book, trades)
        entropy = self.calculate_entropy(order_book)
        pressure = self.calculate_pressure(order_book)
        volume = self.calculate_volume(order_book)
        free_energy = self.calculate_free_energy(temperature, entropy)
        
        return ThermodynamicState(
            temperature=temperature,
            entropy=entropy,
            free_energy=free_energy,
            pressure=pressure,
            volume=volume,
            timestamp=pd.Timestamp.now()
        )
    
    def detect_phase_transition(self, 
                               states: List[ThermodynamicState]) -> bool:
        """
        Detect phase transitions (regime changes)
        
        Phase transition occurs when thermodynamic properties
        change rapidly
        
        Args:
            states: Historical thermodynamic states
            
        Returns:
            True if phase transition detected
        """
        if len(states) < 10:
            return False
        
        recent_states = states[-10:]
        
        # Calculate rate of change
        temp_changes = np.diff([s.temperature for s in recent_states])
        entropy_changes = np.diff([s.entropy for s in recent_states])
        
        # Detect rapid changes
        temp_volatility = np.std(temp_changes)
        entropy_volatility = np.std(entropy_changes)
        
        # Threshold for phase transition
        threshold = 2.0
        
        if temp_volatility > threshold or entropy_volatility > threshold:
            return True
        
        return False


class StatisticalPhysicsAnalyzer:
    """
    Advanced statistical physics analysis
    Based on "An Empirical Analysis of Financial Markets: An Econophysics Approach"
    """
    
    def __init__(self):
        self.returns_history = []
        
    def calculate_power_law_exponent(self, returns: np.ndarray) -> float:
        """
        Calculate power-law exponent for return distribution
        
        P(r) ~ r^(-α)
        
        Fat tails indicate α < 3
        
        Args:
            returns: Array of returns
            
        Returns:
            Power-law exponent α
        """
        abs_returns = np.abs(returns[returns != 0])
        
        if len(abs_returns) < 10:
            return 0.0
        
        # Sort returns
        sorted_returns = np.sort(abs_returns)[::-1]
        
        # Rank
        ranks = np.arange(1, len(sorted_returns) + 1)
        
        # Log-log regression
        log_returns = np.log(sorted_returns)
        log_ranks = np.log(ranks)
        
        # Linear fit
        coeffs = np.polyfit(log_returns, log_ranks, 1)
        alpha = -coeffs[0]
        
        return float(alpha)
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent for long-range dependence
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            prices: Price series
            
        Returns:
            Hurst exponent
        """
        if len(prices) < 20:
            return 0.5
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate R/S statistic
        lags = range(2, min(len(returns) // 2, 100))
        rs_values = []
        
        for lag in lags:
            # Split into chunks
            chunks = [returns[i:i+lag] for i in range(0, len(returns), lag)]
            
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < lag:
                    continue
                    
                # Mean-adjusted series
                mean_adj = chunk - np.mean(chunk)
                
                # Cumulative sum
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(chunk)
                
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if not rs_values:
            return 0.5
        
        # Log-log regression
        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]
        
        return float(np.clip(hurst, 0, 1))
    
    def calculate_correlation_dimension(self, 
                                       prices: np.ndarray,
                                       embedding_dim: int = 5) -> float:
        """
        Calculate correlation dimension (measure of complexity)
        
        Args:
            prices: Price series
            embedding_dim: Embedding dimension
            
        Returns:
            Correlation dimension
        """
        if len(prices) < embedding_dim * 10:
            return 0.0
        
        # Create embedded vectors
        vectors = []
        for i in range(len(prices) - embedding_dim):
            vectors.append(prices[i:i+embedding_dim])
        
        vectors = np.array(vectors)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(vectors))
        
        # Calculate correlation integral for different radii
        radii = np.logspace(-2, 0, 20) * np.std(distances)
        correlations = []
        
        for r in radii:
            count = np.sum(distances < r) - len(vectors)  # Exclude self
            correlation = count / (len(vectors) * (len(vectors) - 1))
            correlations.append(correlation)
        
        # Log-log regression
        log_radii = np.log(radii[np.array(correlations) > 0])
        log_corr = np.log(np.array(correlations)[np.array(correlations) > 0])
        
        if len(log_radii) < 2:
            return 0.0
        
        coeffs = np.polyfit(log_radii, log_corr, 1)
        dimension = coeffs[0]
        
        return float(dimension)
