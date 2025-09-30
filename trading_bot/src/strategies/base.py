# src/strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
import logging

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength from 0 to 1
    price: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    entry_time: datetime
    pnl: Decimal
    pnl_percentage: float

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[Signal] = []
        self.is_active = True
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def should_enter_position(self, signal: Signal, current_positions: Dict[str, Position]) -> bool:
        """Determine if should enter a position based on signal"""
        pass
    
    @abstractmethod
    def should_exit_position(self, position: Position, current_data: pd.Series) -> bool:
        """Determine if should exit a position"""
        pass
    
    def update_positions(self, current_prices: Dict[str, Decimal]):
        """Update position PnL with current prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                
                if position.side == 'long':
                    position.pnl = (position.current_price - position.entry_price) * position.quantity
                else:  # short
                    position.pnl = (position.entry_price - position.current_price) * position.quantity
                
                position.pnl_percentage = float(position.pnl / (position.entry_price * position.quantity)) * 100
    
    def add_position(self, symbol: str, side: str, entry_price: Decimal, quantity: Decimal):
        """Add a new position"""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            pnl=Decimal('0'),
            pnl_percentage=0.0
        )
    
    def close_position(self, symbol: str) -> Optional[Position]:
        """Close and return a position"""
        return self.positions.pop(symbol, None)
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        winning_positions = sum(1 for pos in self.positions.values() if pos.pnl > 0)
        total_positions = len(self.positions)
        
        return {
            'name': self.name,
            'active_positions': total_positions,
            'total_pnl': float(total_pnl),
            'winning_positions': winning_positions,
            'win_rate': winning_positions / total_positions if total_positions > 0 else 0,
            'signals_generated': len(self.signals_history)
        }
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate a signal before processing"""
        if signal.strength < 0 or signal.strength > 1:
            self.logger.warning(f"Invalid signal strength: {signal.strength}")
            return False
        
        if signal.price <= 0:
            self.logger.warning(f"Invalid signal price: {signal.price}")
            return False
        
        return True
    
    def calculate_position_risk(self, signal: Signal, portfolio_value: Decimal) -> Dict[str, float]:
        """Calculate risk metrics for a potential position"""
        # This is a simplified risk calculation
        # In practice, this would be more sophisticated
        estimated_stop_loss = float(signal.price) * 0.05  # 5% stop loss
        max_loss = estimated_stop_loss * float(portfolio_value) * 0.02  # 2% portfolio risk
        
        return {
            'estimated_stop_loss': estimated_stop_loss,
            'max_portfolio_risk': 0.02,
            'estimated_max_loss': max_loss
        }
    
    def get_signal_summary(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent signals"""
        cutoff_time = datetime.now() - pd.Timedelta(hours=lookback_hours)
        recent_signals = [s for s in self.signals_history if s.timestamp >= cutoff_time]
        
        if not recent_signals:
            return {'total_signals': 0}
        
        buy_signals = sum(1 for s in recent_signals if s.signal_type == SignalType.BUY)
        sell_signals = sum(1 for s in recent_signals if s.signal_type == SignalType.SELL)
        avg_strength = np.mean([s.strength for s in recent_signals])
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_signal_strength': float(avg_strength),
            'lookback_hours': lookback_hours
        }
    
    def backtest_signal_accuracy(self, data: pd.DataFrame, forward_periods: int = 5) -> Dict[str, float]:
        """Backtest historical signal accuracy"""
        if len(self.signals_history) < 10:
            return {'insufficient_data': True}
        
        correct_predictions = 0
        total_predictions = 0
        
        for signal in self.signals_history[-100:]:  # Last 100 signals
            # Find corresponding price data
            signal_time = signal.timestamp
            try:
                future_data = data[data.index > signal_time].head(forward_periods)
                if len(future_data) < forward_periods:
                    continue
                
                entry_price = float(signal.price)
                exit_price = float(future_data.iloc[-1]['close'])
                price_change = (exit_price - entry_price) / entry_price
                
                # Check if signal was correct
                if signal.signal_type == SignalType.BUY and price_change > 0:
                    correct_predictions += 1
                elif signal.signal_type == SignalType.SELL and price_change < 0:
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                self.logger.debug(f"Error in backtest calculation: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'forward_periods': forward_periods
        }
    
    def optimize_parameters(self, data: pd.DataFrame, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        best_score = float('-inf')
        best_params = {}
        
        # This is a simplified parameter optimization
        # In practice, you'd use more sophisticated methods
        
        param_keys = list(param_ranges.keys())
        if not param_keys:
            return {'error': 'No parameters to optimize'}
        
        # Simple grid search over first parameter only (for demonstration)
        first_param = param_keys[0]
        original_value = self.config.get(first_param)
        
        for value in param_ranges[first_param]:
            # Temporarily set parameter
            self.config[first_param] = value
            
            # Generate signals with new parameter
            signals = self.generate_signals(data)
            
            # Simple scoring based on signal strength and frequency
            if signals:
                avg_strength = np.mean([s.strength for s in signals])
                signal_frequency = len(signals) / len(data)
                score = avg_strength * signal_frequency
                
                if score > best_score:
                    best_score = score
                    best_params = {first_param: value}
        
        # Restore original value
        self.config[first_param] = original_value
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_completed': True
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.positions:
            return {'no_positions': True}
        
        pnls = [float(pos.pnl) for pos in self.positions.values()]
        pnl_percentages = [pos.pnl_percentage for pos in self.positions.values()]
        
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        pnl_std = np.std(pnls)
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        # Sharpe-like ratio (simplified)
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(pnls),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def reset_strategy(self):
        """Reset strategy state"""
        self.positions.clear()
        self.signals_history.clear()
        self.logger.info(f"Strategy {self.name} has been reset")
    
    def set_active(self, active: bool):
        """Enable or disable strategy"""
        self.is_active = active
        status = "activated" if active else "deactivated"
        self.logger.info(f"Strategy {self.name} has been {status}")

# Example usage
if __name__ == "__main__":
    # This would normally be implemented by concrete strategy classes
    pass