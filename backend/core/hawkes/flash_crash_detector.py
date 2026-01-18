"""
Classification of Flash Crashes Using the Hawkes (p,q) Framework
Implementation of self-exciting point process for flash crash detection

Hawkes processes model self-exciting behavior where past events
increase the probability of future events (cascades).
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.optimize import minimize
import pandas as pd


@dataclass
class HawkesParameters:
    """Parameters for Hawkes (p,q) process"""
    mu: float  # Background intensity
    alpha: np.ndarray  # Excitation coefficients
    beta: np.ndarray  # Decay rates
    p: int  # Number of past events affecting intensity
    q: int  # Number of exponential kernels
    
    @property
    def branching_ratio(self) -> float:
        """
        Branching ratio = Σ(α_i / β_i)
        
        If > 1: Supercritical (explosive)
        If < 1: Subcritical (stable)
        If ≈ 1: Critical (flash crash regime)
        """
        return float(np.sum(self.alpha / self.beta))
    
    def is_explosive(self) -> bool:
        """Check if process is in explosive regime"""
        return self.branching_ratio >= 0.95


@dataclass
class FlashCrashEvent:
    """Detected flash crash event"""
    timestamp: pd.Timestamp
    severity: float  # 0-1 scale
    price_drop: float  # Percentage
    recovery_time: float  # Seconds
    hawkes_intensity: float
    branching_ratio: float
    classification: str  # MINOR, MODERATE, SEVERE, EXTREME


class HawkesProcess:
    """
    Hawkes (p,q) process implementation
    
    λ(t) = μ + Σ Σ α_j * exp(-β_j * (t - t_i))
    
    where:
    - λ(t) = intensity at time t
    - μ = background rate
    - α_j = excitation coefficient
    - β_j = decay rate
    - t_i = past event times
    """
    
    def __init__(self, p: int = 5, q: int = 2):
        """
        Args:
            p: Number of past events to consider
            q: Number of exponential kernels
        """
        self.p = p
        self.q = q
        self.params = None
        self.event_times = []
        
    def fit(self, event_times: np.ndarray) -> HawkesParameters:
        """
        Fit Hawkes process to event data using MLE
        
        Args:
            event_times: Array of event timestamps
            
        Returns:
            Fitted HawkesParameters
        """
        if len(event_times) < 10:
            # Not enough data, return default params
            return HawkesParameters(
                mu=0.1,
                alpha=np.array([0.5] * self.q),
                beta=np.array([1.0] * self.q),
                p=self.p,
                q=self.q
            )
        
        # Sort event times
        event_times = np.sort(event_times)
        self.event_times = event_times
        
        # Initial parameter guess
        T = event_times[-1] - event_times[0]
        n = len(event_times)
        mu_init = n / T
        
        # Initialize alpha and beta
        alpha_init = np.random.uniform(0.1, 0.5, self.q)
        beta_init = np.random.uniform(0.5, 2.0, self.q)
        
        # Combine parameters
        params_init = np.concatenate([[mu_init], alpha_init, beta_init])
        
        # Bounds
        bounds = [(1e-6, None)]  # mu > 0
        bounds.extend([(0, 1)] * self.q)  # 0 < alpha < 1
        bounds.extend([(1e-6, None)] * self.q)  # beta > 0
        
        # Optimize using MLE
        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(event_times,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract parameters
        mu = result.x[0]
        alpha = result.x[1:1+self.q]
        beta = result.x[1+self.q:]
        
        self.params = HawkesParameters(
            mu=mu,
            alpha=alpha,
            beta=beta,
            p=self.p,
            q=self.q
        )
        
        return self.params
    
    def _negative_log_likelihood(self, 
                                params: np.ndarray,
                                event_times: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for MLE
        
        L = Σ log(λ(t_i)) - ∫ λ(t) dt
        """
        mu = params[0]
        alpha = params[1:1+self.q]
        beta = params[1+self.q:]
        
        n = len(event_times)
        T = event_times[-1] - event_times[0]
        
        # First term: Σ log(λ(t_i))
        log_likelihood = 0.0
        
        for i in range(n):
            intensity = self._calculate_intensity(
                event_times[i],
                event_times[:i],
                mu, alpha, beta
            )
            if intensity > 0:
                log_likelihood += np.log(intensity)
        
        # Second term: ∫ λ(t) dt
        integral = mu * T
        
        for i in range(n):
            for j in range(self.q):
                integral += alpha[j] / beta[j] * (
                    1 - np.exp(-beta[j] * (T - (event_times[i] - event_times[0])))
                )
        
        return -(log_likelihood - integral)
    
    def _calculate_intensity(self,
                           t: float,
                           past_events: np.ndarray,
                           mu: float,
                           alpha: np.ndarray,
                           beta: np.ndarray) -> float:
        """
        Calculate intensity λ(t) at time t
        
        λ(t) = μ + Σ Σ α_j * exp(-β_j * (t - t_i))
        """
        intensity = mu
        
        if len(past_events) == 0:
            return intensity
        
        # Consider only recent p events
        recent_events = past_events[-self.p:] if len(past_events) > self.p else past_events
        
        for t_i in recent_events:
            dt = t - t_i
            if dt > 0:
                for j in range(self.q):
                    intensity += alpha[j] * np.exp(-beta[j] * dt)
        
        return intensity
    
    def predict_intensity(self, t: float) -> float:
        """
        Predict intensity at time t using fitted parameters
        
        Args:
            t: Time point
            
        Returns:
            Predicted intensity
        """
        if self.params is None:
            return 0.0
        
        # Get past events before t
        past_events = self.event_times[self.event_times < t]
        
        return self._calculate_intensity(
            t,
            past_events,
            self.params.mu,
            self.params.alpha,
            self.params.beta
        )
    
    def simulate(self, T: float, seed: int = None) -> np.ndarray:
        """
        Simulate Hawkes process using Ogata's thinning algorithm
        
        Args:
            T: Time horizon
            seed: Random seed
            
        Returns:
            Array of simulated event times
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self.params is None:
            raise ValueError("Must fit model before simulation")
        
        events = []
        t = 0
        
        while t < T:
            # Upper bound on intensity
            lambda_max = self.predict_intensity(t) + 1.0
            
            # Generate candidate event
            u = np.random.uniform()
            t = t - np.log(u) / lambda_max
            
            if t > T:
                break
            
            # Accept/reject
            lambda_t = self.predict_intensity(t)
            d = np.random.uniform()
            
            if d * lambda_max <= lambda_t:
                events.append(t)
                self.event_times = np.append(self.event_times, t)
        
        return np.array(events)


class FlashCrashDetector:
    """
    Flash crash detection using Hawkes process
    """
    
    def __init__(self, 
                 price_drop_threshold: float = 0.05,
                 intensity_threshold: float = 10.0):
        """
        Args:
            price_drop_threshold: Minimum price drop to consider (5%)
            intensity_threshold: Hawkes intensity threshold
        """
        self.price_drop_threshold = price_drop_threshold
        self.intensity_threshold = intensity_threshold
        self.hawkes = HawkesProcess(p=10, q=3)
        
    def detect_flash_crash(self,
                          prices: np.ndarray,
                          timestamps: np.ndarray,
                          volumes: np.ndarray) -> List[FlashCrashEvent]:
        """
        Detect flash crashes in price data
        
        Args:
            prices: Price series
            timestamps: Timestamp series
            volumes: Volume series
            
        Returns:
            List of detected flash crash events
        """
        events = []
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Find large negative returns
        crash_indices = np.where(returns < -self.price_drop_threshold)[0]
        
        if len(crash_indices) == 0:
            return events
        
        # Convert to event times (in seconds)
        event_times = timestamps[crash_indices].astype(float) / 1e9
        
        # Fit Hawkes process
        params = self.hawkes.fit(event_times)
        
        # Analyze each crash
        for idx in crash_indices:
            t = timestamps[idx]
            
            # Calculate intensity at crash time
            intensity = self.hawkes.predict_intensity(float(t) / 1e9)
            
            # Calculate price drop
            price_drop = abs(returns[idx])
            
            # Find recovery time
            recovery_time = self._calculate_recovery_time(
                prices, idx, prices[idx]
            )
            
            # Calculate severity
            severity = self._calculate_severity(
                price_drop,
                intensity,
                params.branching_ratio,
                recovery_time
            )
            
            # Classify
            classification = self._classify_crash(severity)
            
            event = FlashCrashEvent(
                timestamp=pd.Timestamp(t),
                severity=severity,
                price_drop=price_drop * 100,
                recovery_time=recovery_time,
                hawkes_intensity=intensity,
                branching_ratio=params.branching_ratio,
                classification=classification
            )
            
            events.append(event)
        
        return events
    
    def _calculate_recovery_time(self,
                                prices: np.ndarray,
                                crash_idx: int,
                                crash_price: float) -> float:
        """
        Calculate time to recover to pre-crash price
        
        Args:
            prices: Price series
            crash_idx: Index of crash
            crash_price: Price at crash
            
        Returns:
            Recovery time in seconds
        """
        pre_crash_price = prices[crash_idx]
        
        # Look forward for recovery
        for i in range(crash_idx + 1, min(crash_idx + 1000, len(prices))):
            if prices[i] >= pre_crash_price * 0.99:  # 99% recovery
                return float(i - crash_idx)
        
        # No recovery found
        return float('inf')
    
    def _calculate_severity(self,
                          price_drop: float,
                          intensity: float,
                          branching_ratio: float,
                          recovery_time: float) -> float:
        """
        Calculate crash severity score [0, 1]
        
        Combines:
        - Price drop magnitude
        - Hawkes intensity
        - Branching ratio
        - Recovery time
        """
        # Normalize components
        price_score = min(price_drop / 0.2, 1.0)  # Max at 20% drop
        intensity_score = min(intensity / 50.0, 1.0)  # Max at 50
        branching_score = min(branching_ratio, 1.0)
        recovery_score = min(recovery_time / 300.0, 1.0)  # Max at 5 min
        
        # Weighted average
        severity = (
            0.4 * price_score +
            0.2 * intensity_score +
            0.2 * branching_score +
            0.2 * recovery_score
        )
        
        return float(severity)
    
    def _classify_crash(self, severity: float) -> str:
        """Classify crash based on severity"""
        if severity < 0.25:
            return "MINOR"
        elif severity < 0.5:
            return "MODERATE"
        elif severity < 0.75:
            return "SEVERE"
        else:
            return "EXTREME"
    
    def predict_crash_probability(self,
                                  current_time: float,
                                  lookback_window: int = 100) -> float:
        """
        Predict probability of flash crash in next period
        
        Args:
            current_time: Current timestamp
            lookback_window: Number of past events to consider
            
        Returns:
            Crash probability [0, 1]
        """
        if self.hawkes.params is None:
            return 0.0
        
        # Get current intensity
        intensity = self.hawkes.predict_intensity(current_time)
        
        # Get branching ratio
        branching_ratio = self.hawkes.params.branching_ratio
        
        # High intensity + high branching ratio = high crash risk
        intensity_factor = min(intensity / self.intensity_threshold, 1.0)
        branching_factor = min(branching_ratio, 1.0)
        
        probability = 0.5 * intensity_factor + 0.5 * branching_factor
        
        return float(probability)
