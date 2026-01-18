"""
Portfolio Optimization Module
Implementation of Modern Portfolio Theory and Risk Parity

Based on:
- "Tfin Crypto: From Speculation to Optimization in Risk Managed Crypto Portfolio Allocation"
- Modern Portfolio Theory (Markowitz)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import cvxpy as cp


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    
    def to_dict(self) -> Dict:
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns_covariance(self,
                                    prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate expected returns and covariance matrix
        
        Args:
            prices: DataFrame of historical prices
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Expected returns (annualized)
        expected_returns = returns.mean() * 252
        
        # Covariance matrix (annualized)
        cov_matrix = returns.cov() * 252
        
        return expected_returns.values, cov_matrix.values
    
    def optimize_sharpe(self,
                       expected_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       symbols: List[str]) -> PortfolioAllocation:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            symbols: Asset symbols
            
        Returns:
            PortfolioAllocation object
        """
        num_assets = len(symbols)
        
        # Objective: Minimize negative Sharpe ratio
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds: 0 <= weight <= 0.4 (max 40% per asset)
        bounds = tuple((0, 0.4) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Calculate metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Create weights dictionary
        weights_dict = {symbol: float(w) for symbol, w in zip(symbols, optimal_weights)}
        
        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            max_drawdown=0.0,  # Would calculate from backtest
            var_95=0.0,
            cvar_95=0.0
        )
    
    def optimize_min_variance(self,
                             cov_matrix: np.ndarray,
                             symbols: List[str]) -> PortfolioAllocation:
        """
        Optimize for minimum variance
        
        Args:
            cov_matrix: Covariance matrix
            symbols: Asset symbols
            
        Returns:
            PortfolioAllocation object
        """
        num_assets = len(symbols)
        
        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, 0.4) for _ in range(num_assets))
        
        # Initial guess
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))
        
        weights_dict = {symbol: float(w) for symbol, w in zip(symbols, optimal_weights)}
        
        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=0.0,
            volatility=float(portfolio_vol),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def risk_parity(self,
                   cov_matrix: np.ndarray,
                   symbols: List[str]) -> PortfolioAllocation:
        """
        Risk parity allocation (equal risk contribution)
        
        Each asset contributes equally to portfolio risk
        
        Args:
            cov_matrix: Covariance matrix
            symbols: Asset symbols
            
        Returns:
            PortfolioAllocation object
        """
        num_assets = len(symbols)
        
        # Objective: Minimize difference in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Marginal risk contribution
            mrc = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Risk contribution
            rc = weights * mrc
            
            # Target: equal risk contribution
            target_rc = portfolio_vol / num_assets
            
            # Sum of squared differences
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = tuple((0.01, 0.5) for _ in range(num_assets))
        
        # Initial guess
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        weights_dict = {symbol: float(w) for symbol, w in zip(symbols, optimal_weights)}
        
        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=0.0,
            volatility=float(portfolio_vol),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def kelly_criterion(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing
        
        Kelly% = W - [(1-W) / R]
        
        where:
        - W = win rate
        - R = avg_win / avg_loss
        
        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly percentage
        """
        if avg_loss == 0:
            return 0.0
        
        R = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / R)
        
        # Use fractional Kelly (0.25) for safety
        fractional_kelly = kelly * 0.25
        
        return max(0.0, min(fractional_kelly, 0.3))  # Cap at 30%
    
    def calculate_var_cvar(self,
                          returns: np.ndarray,
                          confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR
        
        Args:
            returns: Historical returns
            confidence: Confidence level (default 95%)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # VaR: percentile
        var_index = int((1 - confidence) * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # CVaR: average of returns below VaR
        cvar = sorted_returns[:var_index].mean()
        
        return float(var), float(cvar)
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown (positive value)
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return float(abs(max_dd))
