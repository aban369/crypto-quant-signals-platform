"""
Comprehensive Signal Generator
Combines all research methodologies into unified trading signals

Integrates:
- Econophysics (temperature, entropy)
- Order book dynamics (OFI, microstructure)
- DeepLOB predictions
- Hawkes process (flash crash detection)
- RL agent recommendations
- Manipulation detection
- Portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import all modules
import sys
sys.path.append('..')

from econophysics.temperature import OrderBookThermodynamics, StatisticalPhysicsAnalyzer
from hawkes.flash_crash_detector import FlashCrashDetector
from orderbook.ofi_calculator import OrderFlowImbalance, EnhancedOFI
from deeplob.model import DeepLOBPredictor
from rl_agents.ppo_trader import PPOCryptoTrader, EnsembleRLTrader
from manipulation.spoofing_detector import ManipulationDetectionSystem
from portfolio.optimizer import PortfolioOptimizer


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0


class SignalDirection(Enum):
    """Signal direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradingSignal:
    """Comprehensive trading signal"""
    timestamp: pd.Timestamp
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float  # 0-1
    
    # Component signals
    econophysics_signal: Dict
    orderbook_signal: Dict
    deeplob_signal: Dict
    hawkes_signal: Dict
    rl_signal: Dict
    manipulation_alert: Optional[Dict]
    
    # Metrics
    expected_return: float
    risk_score: float
    sharpe_estimate: float
    
    # Recommendations
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # Percentage of portfolio
    
    # Metadata
    reasoning: str
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': str(self.timestamp),
            'symbol': self.symbol,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'econophysics': self.econophysics_signal,
            'orderbook': self.orderbook_signal,
            'deeplob': self.deeplob_signal,
            'hawkes': self.hawkes_signal,
            'rl': self.rl_signal,
            'manipulation': self.manipulation_alert,
            'expected_return': self.expected_return,
            'risk_score': self.risk_score,
            'sharpe_estimate': self.sharpe_estimate,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'reasoning': self.reasoning,
            'warnings': self.warnings
        }


class SignalGenerator:
    """
    Master signal generator combining all research methodologies
    """
    
    def __init__(self, symbols: List[str]):
        """
        Args:
            symbols: List of trading symbols
        """
        self.symbols = symbols
        
        # Initialize all components
        self.thermodynamics = OrderBookThermodynamics()
        self.stat_physics = StatisticalPhysicsAnalyzer()
        self.flash_crash_detector = FlashCrashDetector()
        self.ofi_calculator = OrderFlowImbalance()
        self.enhanced_ofi = EnhancedOFI()
        self.deeplob = DeepLOBPredictor()
        self.rl_trader = EnsembleRLTrader(symbols=symbols, num_agents=3)
        self.manipulation_detector = ManipulationDetectionSystem()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Signal history
        self.signal_history = {symbol: [] for symbol in symbols}
        
    def generate_signal(self,
                       symbol: str,
                       orderbook: Dict,
                       trades: List[Dict],
                       orderbook_history: List[Dict],
                       prices: np.ndarray,
                       volumes: np.ndarray,
                       timestamps: np.ndarray) -> TradingSignal:
        """
        Generate comprehensive trading signal
        
        Args:
            symbol: Trading symbol
            orderbook: Current order book
            trades: Recent trades
            orderbook_history: Historical order book snapshots
            prices: Price history
            volumes: Volume history
            timestamps: Timestamp history
            
        Returns:
            TradingSignal object
        """
        warnings = []
        
        # 1. ECONOPHYSICS ANALYSIS
        thermo_state = self.thermodynamics.get_thermodynamic_state(orderbook, trades)
        
        # Calculate statistical physics metrics
        returns = np.diff(prices) / prices[:-1]
        power_law_exp = self.stat_physics.calculate_power_law_exponent(returns)
        hurst_exp = self.stat_physics.calculate_hurst_exponent(prices)
        
        econophysics_signal = {
            'temperature': thermo_state.temperature,
            'entropy': thermo_state.entropy,
            'pressure': thermo_state.pressure,
            'phase': thermo_state.phase(),
            'power_law_exponent': power_law_exp,
            'hurst_exponent': hurst_exp,
            'regime': self._interpret_regime(thermo_state, hurst_exp)
        }
        
        # 2. ORDER BOOK ANALYSIS
        ofi_signal = self.ofi_calculator.generate_signal(orderbook)
        orderbook_features = self.enhanced_ofi.calculate_all_features(orderbook)
        
        orderbook_signal = {
            'ofi_total': ofi_signal.ofi_total,
            'ofi_direction': ofi_signal.predicted_direction,
            'ofi_confidence': ofi_signal.confidence,
            'volume_imbalance': orderbook_features['volume_imbalance'],
            'depth_imbalance': orderbook_features['depth_imbalance'],
            'spread': orderbook_features['spread'],
            'microprice': orderbook_features['microprice']
        }
        
        # 3. DEEPLOB PREDICTION
        deeplob_pred = self.deeplob.predict(orderbook_history[-100:])
        
        deeplob_signal = {
            'direction': deeplob_pred.direction,
            'up_prob': deeplob_pred.up_probability,
            'down_prob': deeplob_pred.down_probability,
            'stationary_prob': deeplob_pred.stationary_probability,
            'confidence': deeplob_pred.confidence
        }
        
        # 4. HAWKES PROCESS (FLASH CRASH DETECTION)
        flash_crashes = self.flash_crash_detector.detect_flash_crash(
            prices, timestamps, volumes
        )
        
        crash_probability = self.flash_crash_detector.predict_crash_probability(
            float(timestamps[-1]) / 1e9
        )
        
        hawkes_signal = {
            'crash_probability': crash_probability,
            'num_crashes_detected': len(flash_crashes),
            'recent_crashes': [c.__dict__ for c in flash_crashes[-3:]] if flash_crashes else []
        }
        
        if crash_probability > 0.7:
            warnings.append(f"HIGH FLASH CRASH RISK: {crash_probability:.1%}")
        
        # 5. RL AGENT RECOMMENDATION
        # Create state vector (simplified)
        state = self._create_state_vector(
            orderbook, prices, volumes, thermo_state, ofi_signal
        )
        
        rl_actions = self.rl_trader.predict_ensemble(state)
        rl_action = next((a for a in rl_actions if a.symbol == symbol), None)
        
        rl_signal = {
            'action': rl_action.action if rl_action else "HOLD",
            'position_size': rl_action.position_size if rl_action else 0.0,
            'confidence': rl_action.confidence if rl_action else 0.0,
            'expected_return': rl_action.expected_return if rl_action else 0.0
        }
        
        # 6. MANIPULATION DETECTION
        manipulation_alerts = self.manipulation_detector.detect_all(
            orderbook=orderbook,
            trades=trades,
            orders=[],  # Would need real order data
            cancels=[],  # Would need real cancel data
            current_price=prices[-1],
            current_volume=volumes[-1]
        )
        
        manipulation_alert = None
        if manipulation_alerts:
            # Take highest severity alert
            top_alert = max(manipulation_alerts, key=lambda x: x.severity)
            manipulation_alert = top_alert.to_dict()
            warnings.append(f"MANIPULATION DETECTED: {top_alert.manipulation_type}")
        
        # 7. AGGREGATE SIGNALS
        direction, strength, confidence = self._aggregate_signals(
            econophysics_signal,
            orderbook_signal,
            deeplob_signal,
            hawkes_signal,
            rl_signal,
            manipulation_alert
        )
        
        # 8. RISK MANAGEMENT
        current_price = prices[-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        
        entry_price, stop_loss, take_profit = self._calculate_levels(
            current_price, direction, volatility
        )
        
        # Position sizing using Kelly Criterion
        win_rate = 0.55  # Estimated from backtest
        avg_win = 0.03  # 3% average win
        avg_loss = 0.015  # 1.5% average loss
        
        kelly_size = self.portfolio_optimizer.kelly_criterion(
            win_rate, avg_win, avg_loss
        )
        
        position_size = kelly_size * confidence
        
        # Risk score
        risk_score = self._calculate_risk_score(
            volatility, crash_probability, manipulation_alert, thermo_state
        )
        
        # Expected return
        expected_return = self._estimate_return(
            direction, strength, confidence, volatility
        )
        
        # Sharpe estimate
        sharpe_estimate = expected_return / volatility if volatility > 0 else 0
        
        # Reasoning
        reasoning = self._generate_reasoning(
            econophysics_signal, orderbook_signal, deeplob_signal,
            hawkes_signal, rl_signal, direction, strength
        )
        
        # Create signal
        signal = TradingSignal(
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            econophysics_signal=econophysics_signal,
            orderbook_signal=orderbook_signal,
            deeplob_signal=deeplob_signal,
            hawkes_signal=hawkes_signal,
            rl_signal=rl_signal,
            manipulation_alert=manipulation_alert,
            expected_return=expected_return,
            risk_score=risk_score,
            sharpe_estimate=sharpe_estimate,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=reasoning,
            warnings=warnings
        )
        
        # Store in history
        self.signal_history[symbol].append(signal)
        
        return signal
    
    def _interpret_regime(self, thermo_state, hurst_exp: float) -> str:
        """Interpret market regime"""
        phase = thermo_state.phase()
        
        if hurst_exp > 0.6:
            trend = "TRENDING"
        elif hurst_exp < 0.4:
            trend = "MEAN_REVERTING"
        else:
            trend = "RANDOM_WALK"
        
        return f"{phase}_{trend}"
    
    def _create_state_vector(self, orderbook, prices, volumes, 
                            thermo_state, ofi_signal) -> np.ndarray:
        """Create state vector for RL agent"""
        # Simplified state vector
        state = np.zeros(100)  # Placeholder
        return state
    
    def _aggregate_signals(self, econophysics, orderbook, deeplob,
                          hawkes, rl, manipulation) -> Tuple:
        """Aggregate all signals into final direction and strength"""
        
        # Voting system with weights
        votes = {
            'LONG': 0.0,
            'SHORT': 0.0,
            'NEUTRAL': 0.0
        }
        
        # Econophysics vote (weight: 0.15)
        if econophysics['temperature'] > 1.5 and econophysics['pressure'] > 0.3:
            votes['LONG'] += 0.15
        elif econophysics['temperature'] > 1.5 and econophysics['pressure'] < -0.3:
            votes['SHORT'] += 0.15
        else:
            votes['NEUTRAL'] += 0.15
        
        # Order book vote (weight: 0.25)
        if orderbook['ofi_direction'] == 'UP':
            votes['LONG'] += 0.25 * orderbook['ofi_confidence']
        elif orderbook['ofi_direction'] == 'DOWN':
            votes['SHORT'] += 0.25 * orderbook['ofi_confidence']
        else:
            votes['NEUTRAL'] += 0.25
        
        # DeepLOB vote (weight: 0.30)
        if deeplob['direction'] == 'UP':
            votes['LONG'] += 0.30 * deeplob['confidence']
        elif deeplob['direction'] == 'DOWN':
            votes['SHORT'] += 0.30 * deeplob['confidence']
        else:
            votes['NEUTRAL'] += 0.30
        
        # RL agent vote (weight: 0.20)
        if rl['action'] == 'BUY':
            votes['LONG'] += 0.20 * rl['confidence']
        elif rl['action'] == 'SELL':
            votes['SHORT'] += 0.20 * rl['confidence']
        else:
            votes['NEUTRAL'] += 0.20
        
        # Hawkes penalty (weight: 0.10)
        if hawkes['crash_probability'] > 0.5:
            # Reduce all non-neutral votes
            votes['LONG'] *= (1 - hawkes['crash_probability'] * 0.5)
            votes['SHORT'] *= (1 - hawkes['crash_probability'] * 0.5)
            votes['NEUTRAL'] += 0.10
        
        # Manipulation penalty
        if manipulation and manipulation['severity'] > 0.7:
            votes['NEUTRAL'] += 0.20
            votes['LONG'] *= 0.5
            votes['SHORT'] *= 0.5
        
        # Determine direction
        max_vote = max(votes.values())
        direction_str = max(votes, key=votes.get)
        
        direction = SignalDirection[direction_str]
        
        # Determine strength
        if max_vote > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif max_vote > 0.6:
            strength = SignalStrength.STRONG
        elif max_vote > 0.4:
            strength = SignalStrength.MODERATE
        elif max_vote > 0.2:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        # Confidence
        confidence = max_vote
        
        return direction, strength, confidence
    
    def _calculate_levels(self, current_price, direction, volatility):
        """Calculate entry, stop loss, and take profit levels"""
        entry_price = current_price
        
        if direction == SignalDirection.LONG:
            stop_loss = current_price * (1 - 2 * volatility)
            take_profit = current_price * (1 + 4 * volatility)
        elif direction == SignalDirection.SHORT:
            stop_loss = current_price * (1 + 2 * volatility)
            take_profit = current_price * (1 - 4 * volatility)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        return entry_price, stop_loss, take_profit
    
    def _calculate_risk_score(self, volatility, crash_prob, 
                             manipulation, thermo_state) -> float:
        """Calculate overall risk score"""
        risk = 0.0
        
        # Volatility risk
        risk += min(volatility / 0.05, 1.0) * 0.3
        
        # Flash crash risk
        risk += crash_prob * 0.3
        
        # Manipulation risk
        if manipulation:
            risk += manipulation['severity'] * 0.2
        
        # Thermodynamic risk (high entropy = high risk)
        risk += thermo_state.entropy * 0.2
        
        return min(risk, 1.0)
    
    def _estimate_return(self, direction, strength, confidence, volatility):
        """Estimate expected return"""
        if direction == SignalDirection.NEUTRAL:
            return 0.0
        
        base_return = volatility * 2  # 2x volatility
        strength_multiplier = strength.value / 5.0
        
        expected_return = base_return * strength_multiplier * confidence
        
        return expected_return
    
    def _generate_reasoning(self, econophysics, orderbook, deeplob,
                           hawkes, rl, direction, strength) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        # Econophysics
        phase = econophysics['phase']
        reasons.append(f"Market phase: {phase}")
        
        # Order book
        if orderbook['ofi_confidence'] > 0.6:
            reasons.append(f"Strong order flow imbalance: {orderbook['ofi_direction']}")
        
        # DeepLOB
        if deeplob['confidence'] > 0.7:
            reasons.append(f"DeepLOB predicts {deeplob['direction']} with {deeplob['confidence']:.1%} confidence")
        
        # RL
        if rl['confidence'] > 0.6:
            reasons.append(f"RL agent recommends {rl['action']}")
        
        # Hawkes
        if hawkes['crash_probability'] > 0.5:
            reasons.append(f"⚠️ Flash crash risk: {hawkes['crash_probability']:.1%}")
        
        reasoning = " | ".join(reasons)
        
        return f"{direction.value} {strength.name}: {reasoning}"
