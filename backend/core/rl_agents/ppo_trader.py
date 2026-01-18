"""
Cryptocurrency Futures Portfolio Trading System Using Reinforcement Learning
Implementation of PPO (Proximal Policy Optimization) for crypto trading

Based on:
- "FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading"
- "Tfin Crypto: From Speculation to Optimization in Risk Managed Crypto Portfolio Allocation"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces


@dataclass
class TradingAction:
    """Trading action output"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    position_size: float  # Percentage of portfolio
    confidence: float
    expected_return: float
    risk_score: float


class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for crypto trading
    
    State space:
    - Price features (OHLCV)
    - Technical indicators
    - Order book features
    - Portfolio state
    - Market regime
    
    Action space:
    - Continuous: [-1, 1] for each asset
      -1 = full short, 0 = neutral, 1 = full long
    """
    
    def __init__(self,
                 symbols: List[str],
                 initial_balance: float = 100000,
                 max_position_size: float = 0.3,
                 transaction_cost: float = 0.001):
        """
        Args:
            symbols: List of trading symbols
            initial_balance: Initial portfolio balance
            max_position_size: Maximum position size per asset
            transaction_cost: Transaction cost (0.1%)
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.symbols = symbols
        self.num_assets = len(symbols)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        
        # State space
        # Features per asset: OHLCV (5) + indicators (10) + orderbook (5) = 20
        # Portfolio: balance (1) + positions (num_assets) + PnL (1) = num_assets + 2
        # Market: regime (1) + volatility (1) = 2
        state_dim = self.num_assets * 20 + self.num_assets + 2 + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: continuous [-1, 1] for each asset
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.total_pnl = 0.0
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step
        
        Args:
            action: Array of actions for each asset
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Execute trades
        pnl = self._execute_trades(action)
        
        # Calculate reward
        reward = self._calculate_reward(pnl, action)
        
        # Update state
        self.step_count += 1
        state = self._get_state()
        
        # Check if done
        done = self._is_done()
        
        # Info
        info = {
            'pnl': pnl,
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'positions': self.positions.copy()
        }
        
        return state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state
        
        Returns:
            State vector
        """
        state = []
        
        # Asset features (placeholder - would be filled with real data)
        for symbol in self.symbols:
            # OHLCV
            state.extend([0.0] * 5)
            # Technical indicators
            state.extend([0.0] * 10)
            # Order book features
            state.extend([0.0] * 5)
        
        # Portfolio state
        state.append(self.balance / self.initial_balance)  # Normalized balance
        for symbol in self.symbols:
            state.append(self.positions[symbol])
        state.append(self.total_pnl / self.initial_balance)  # Normalized PnL
        
        # Market state
        state.append(0.0)  # Market regime
        state.append(0.0)  # Volatility
        
        return np.array(state, dtype=np.float32)
    
    def _execute_trades(self, action: np.ndarray) -> float:
        """
        Execute trades based on action
        
        Args:
            action: Action vector
            
        Returns:
            PnL from trades
        """
        pnl = 0.0
        
        for i, symbol in enumerate(self.symbols):
            target_position = action[i] * self.max_position_size
            current_position = self.positions[symbol]
            
            # Calculate position change
            position_change = target_position - current_position
            
            if abs(position_change) > 0.01:  # Minimum trade size
                # Execute trade
                # (Placeholder - would use real price data)
                current_price = 100.0
                
                # Calculate cost
                trade_value = abs(position_change) * self.balance
                cost = trade_value * self.transaction_cost
                
                # Update position
                self.positions[symbol] = target_position
                self.entry_prices[symbol] = current_price
                
                # Deduct cost
                self.balance -= cost
                pnl -= cost
        
        return pnl
    
    def _calculate_reward(self, pnl: float, action: np.ndarray) -> float:
        """
        Calculate reward (risk-adjusted returns)
        
        Reward = Returns - λ * Risk
        
        where:
        - Returns = PnL
        - Risk = Volatility of returns + Drawdown
        - λ = Risk aversion parameter
        
        Args:
            pnl: Profit/Loss
            action: Action taken
            
        Returns:
            Reward value
        """
        # Returns component
        returns = pnl / self.initial_balance
        
        # Risk component
        # Portfolio volatility (simplified)
        portfolio_vol = np.std(action) * 0.1
        
        # Drawdown penalty
        drawdown = max(0, -self.total_pnl / self.initial_balance)
        
        # Risk aversion parameter
        lambda_risk = 0.5
        
        # Risk-adjusted reward
        reward = returns - lambda_risk * (portfolio_vol + drawdown)
        
        # Sharpe ratio bonus
        if portfolio_vol > 0:
            sharpe = returns / portfolio_vol
            reward += 0.1 * sharpe
        
        return float(reward)
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Done if balance drops below 50% or max steps reached
        if self.balance < self.initial_balance * 0.5:
            return True
        if self.step_count >= 1000:
            return True
        return False


class PPOCryptoTrader:
    """
    PPO-based crypto trading agent
    """
    
    def __init__(self,
                 symbols: List[str],
                 model_path: str = None):
        """
        Args:
            symbols: List of trading symbols
            model_path: Path to saved model
        """
        self.symbols = symbols
        
        # Create environment
        env = CryptoTradingEnv(symbols=symbols)
        self.env = DummyVecEnv([lambda: env])
        
        # Create PPO model
        if model_path:
            self.model = PPO.load(model_path, env=self.env)
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1
            )
    
    def train(self, total_timesteps: int = 100000):
        """
        Train the agent
        
        Args:
            total_timesteps: Total training steps
        """
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, state: np.ndarray) -> List[TradingAction]:
        """
        Predict trading actions
        
        Args:
            state: Current state
            
        Returns:
            List of TradingAction objects
        """
        action, _ = self.model.predict(state, deterministic=True)
        
        actions = []
        for i, symbol in enumerate(self.symbols):
            action_value = float(action[0][i])
            
            # Determine action type
            if action_value > 0.1:
                action_type = "BUY"
            elif action_value < -0.1:
                action_type = "SELL"
            else:
                action_type = "HOLD"
            
            # Position size
            position_size = abs(action_value) * 0.3  # Max 30%
            
            # Confidence (based on magnitude)
            confidence = min(abs(action_value), 1.0)
            
            trading_action = TradingAction(
                symbol=symbol,
                action=action_type,
                position_size=position_size,
                confidence=confidence,
                expected_return=action_value * 0.05,  # Estimated
                risk_score=1.0 - confidence
            )
            
            actions.append(trading_action)
        
        return actions
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model"""
        self.model = PPO.load(path, env=self.env)


class EnsembleRLTrader:
    """
    Ensemble of RL agents for robust trading
    Based on "FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning"
    """
    
    def __init__(self, 
                 symbols: List[str],
                 num_agents: int = 5):
        """
        Args:
            symbols: Trading symbols
            num_agents: Number of agents in ensemble
        """
        self.symbols = symbols
        self.num_agents = num_agents
        
        # Create ensemble of agents
        self.agents = [
            PPOCryptoTrader(symbols=symbols)
            for _ in range(num_agents)
        ]
    
    def train_ensemble(self, total_timesteps: int = 100000):
        """Train all agents in ensemble"""
        for i, agent in enumerate(self.agents):
            print(f"Training agent {i+1}/{self.num_agents}")
            agent.train(total_timesteps)
    
    def predict_ensemble(self, state: np.ndarray) -> List[TradingAction]:
        """
        Predict using ensemble (voting)
        
        Args:
            state: Current state
            
        Returns:
            Aggregated trading actions
        """
        # Get predictions from all agents
        all_predictions = []
        for agent in self.agents:
            predictions = agent.predict(state)
            all_predictions.append(predictions)
        
        # Aggregate predictions
        aggregated_actions = []
        
        for i, symbol in enumerate(self.symbols):
            # Collect actions for this symbol
            symbol_actions = [pred[i] for pred in all_predictions]
            
            # Vote on action type
            action_votes = [a.action for a in symbol_actions]
            action_type = max(set(action_votes), key=action_votes.count)
            
            # Average position size and confidence
            avg_position_size = np.mean([a.position_size for a in symbol_actions])
            avg_confidence = np.mean([a.confidence for a in symbol_actions])
            avg_expected_return = np.mean([a.expected_return for a in symbol_actions])
            avg_risk_score = np.mean([a.risk_score for a in symbol_actions])
            
            aggregated_action = TradingAction(
                symbol=symbol,
                action=action_type,
                position_size=avg_position_size,
                confidence=avg_confidence,
                expected_return=avg_expected_return,
                risk_score=avg_risk_score
            )
            
            aggregated_actions.append(aggregated_action)
        
        return aggregated_actions
