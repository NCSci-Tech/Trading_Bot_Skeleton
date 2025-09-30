# src/strategies/momentum.py
from .base import BaseStrategy, Signal, SignalType, Position
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy using RSI and Moving Averages"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MomentumStrategy", config)
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.ma_short_period = config.get('ma_short', 10)
        self.ma_long_period = config.get('ma_long', 20)
        self.volume_threshold = config.get('volume_threshold', 1.2)  # Volume must be 1.2x average
        
        # Risk parameters
        self.stop_loss_pct = config.get('stop_loss_percentage', 0.05)  # 5%
        self.take_profit_pct = config.get('take_profit_percentage', 0.10)  # 10%
        self.max_position_size = Decimal(str(config.get('max_position_size', '0.1')))
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate momentum-based trading signals"""
        if len(data) < max(self.rsi_period, self.ma_long_period):
            return []
        
        signals = []
        latest = data.iloc[-1]
        symbol = latest.name if hasattr(latest, 'name') else 'UNKNOWN'
        
        # Get indicators
        rsi = latest.get('rsi')
        sma_short = latest.get(f'sma_{self.ma_short_period}')
        sma_long = latest.get(f'sma_{self.ma_long_period}')
        volume_ratio = latest.get('volume_ratio', 1.0)
        close_price = latest['close']
        
        if pd.isna(rsi) or pd.isna(sma_short) or pd.isna(sma_long):
            return signals
        
        # Calculate signal strength
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        
        # Bullish momentum conditions
        if (rsi < self.rsi_oversold and 
            sma_short > sma_long and 
            volume_ratio > self.volume_threshold):
            
            signal_type = SignalType.BUY
            signal_strength = min(1.0, (self.rsi_oversold - rsi) / 10 + 
                                (sma_short - sma_long) / sma_long + 
                                min(0.3, (volume_ratio - 1.0)))
        
        # Bearish momentum conditions
        elif (rsi > self.rsi_overbought and 
              sma_short < sma_long and 
              volume_ratio > self.volume_threshold):
            
            signal_type = SignalType.SELL
            signal_strength = min(1.0, (rsi - self.rsi_overbought) / 10 + 
                                (sma_long - sma_short) / sma_long + 
                                min(0.3, (volume_ratio - 1.0)))
        
        # Strong upward momentum (price above both MAs)
        elif (close_price > sma_short > sma_long and 
              rsi > 50 and rsi < self.rsi_overbought and
              volume_ratio > 1.0):
            
            signal_type = SignalType.BUY
            signal_strength = min(0.7, (rsi - 50) / 20 + 
                                (close_price - sma_long) / sma_long)
        
        # Strong downward momentum (price below both MAs)
        elif (close_price < sma_short < sma_long and 
              rsi < 50 and rsi > self.rsi_oversold and
              volume_ratio > 1.0):
            
            signal_type = SignalType.SELL
            signal_strength = min(0.7, (50 - rsi) / 20 + 
                                (sma_long - close_price) / sma_long)
        
        # Create signal if strength is sufficient
        if signal_strength > 0.3:  # Minimum signal strength threshold
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=Decimal(str(close_price)),
                timestamp=datetime.now(),
                metadata={
                    'rsi': rsi,
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'volume_ratio': volume_ratio,
                    'strategy_type': 'momentum'
                }
            )
            
            if self.validate_signal(signal):
                signals.append(signal)
                self.signals_history.append(signal)
        
        return signals
    
    def should_enter_position(self, signal: Signal, current_positions: Dict[str, Position]) -> bool:
        """Determine if should enter a position based on signal"""
        # Don't enter if already have position in this symbol
        if signal.symbol in current_positions:
            return False
        
        # Only enter on strong signals
        if signal.strength < 0.5:
            return False
        
        # Only enter buy/sell signals, not holds
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            return True
        
        return False
    
    def should_exit_position(self, position: Position, current_data: pd.Series) -> bool:
        """Determine if should exit a position"""
        # Stop loss check
        if position.pnl_percentage <= -self.stop_loss_pct * 100:
            self.logger.info(f"Stop loss triggered for {position.symbol} at {position.pnl_percentage:.2f}%")
            return True
        
        # Take profit check
        if position.pnl_percentage >= self.take_profit_pct * 100:
            self.logger.info(f"Take profit triggered for {position.symbol} at {position.pnl_percentage:.2f}%")
            return True
        
        # Technical exit conditions based on momentum
        rsi = current_data.get('rsi')
        sma_short = current_data.get(f'sma_{self.ma_short_period}')
        sma_long = current_data.get(f'sma_{self.ma_long_period}')
        close_price = current_data['close']
        
        if pd.notna(rsi) and pd.notna(sma_short) and pd.notna(sma_long):
            # Exit long position on momentum reversal
            if (position.side == 'long' and 
                (rsi > self.rsi_overbought or 
                 (sma_short < sma_long and close_price < sma_short))):
                self.logger.info(f"Momentum reversal exit for long position in {position.symbol}")
                return True
            
            # Exit short position on momentum reversal
            elif (position.side == 'short' and 
                  (rsi < self.rsi_oversold or 
                   (sma_short > sma_long and close_price > sma_short))):
                self.logger.info(f"Momentum reversal exit for short position in {position.symbol}")
                return True
        
        # Time-based exit (hold for maximum 24 hours)
        max_hold_hours = 24
        if (datetime.now() - position.entry_time).total_seconds() > max_hold_hours * 3600:
            self.logger.info(f"Time-based exit for {position.symbol} after {max_hold_hours} hours")
            return True
        
        return False
    
    def analyze_momentum_strength(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze overall momentum strength in the market"""
        if len(data) < self.ma_long_period:
            return {'insufficient_data': True}
        
        latest_data = data.tail(10)  # Last 10 periods
        
        # Calculate momentum indicators
        rsi_values = latest_data['rsi'].dropna()
        ma_short_values = latest_data[f'sma_{self.ma_short_period}'].dropna()
        ma_long_values = latest_data[f'sma_{self.ma_long_period}'].dropna()
        price_changes = latest_data['close'].pct_change().dropna()
        
        if len(rsi_values) == 0 or len(ma_short_values) == 0:
            return {'insufficient_indicators': True}
        
        # Momentum strength metrics
        avg_rsi = float(rsi_values.mean())
        rsi_trend = float(rsi_values.iloc[-1] - rsi_values.iloc[0])
        
        ma_separation = float((ma_short_values.iloc[-1] - ma_long_values.iloc[-1]) / ma_long_values.iloc[-1] * 100)
        
        price_momentum = float(price_changes.mean() * 100)  # Average percentage change
        volatility = float(price_changes.std() * 100)
        
        # Overall momentum score (0 to 1)
        momentum_score = 0.0
        
        # RSI component
        if avg_rsi > 70:
            momentum_score += 0.3  # Strong upward momentum
        elif avg_rsi < 30:
            momentum_score += 0.3  # Strong downward momentum (also momentum)
        else:
            momentum_score += max(0, (abs(avg_rsi - 50) / 50) * 0.3)
        
        # Moving average component
        momentum_score += min(0.4, abs(ma_separation) / 5)
        
        # Price momentum component
        momentum_score += min(0.3, abs(price_momentum) / 2)
        
        return {
            'momentum_score': min(1.0, momentum_score),
            'avg_rsi': avg_rsi,
            'rsi_trend': rsi_trend,
            'ma_separation_pct': ma_separation,
            'price_momentum_pct': price_momentum,
            'volatility_pct': volatility,
            'is_trending': abs(ma_separation) > 1.0,
            'trend_direction': 'up' if ma_separation > 0 else 'down'
        }
    
    def get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization"""
        return {
            'rsi_period': [10, 12, 14, 16, 18, 20],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'ma_short': [5, 8, 10, 12, 15],
            'ma_long': [15, 20, 25, 30],
            'volume_threshold': [1.0, 1.2, 1.5, 2.0],
            'stop_loss_percentage': [0.03, 0.05, 0.08, 0.10],
            'take_profit_percentage': [0.06, 0.10, 0.15, 0.20]
        }
    
    def validate_parameters(self) -> Dict[str, bool]:
        """Validate current strategy parameters"""
        issues = {}
        
        # Check RSI parameters
        if self.rsi_period < 5 or self.rsi_period > 50:
            issues['rsi_period'] = False
        
        if self.rsi_oversold >= self.rsi_overbought:
            issues['rsi_levels'] = False
        
        if self.rsi_oversold < 10 or self.rsi_overbought > 90:
            issues['rsi_extreme_levels'] = False
        
        # Check MA parameters
        if self.ma_short_period >= self.ma_long_period:
            issues['ma_periods'] = False
        
        if self.ma_long_period > 100:
            issues['ma_long_too_large'] = False
        
        # Check risk parameters
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 0.2:
            issues['stop_loss'] = False
        
        if self.take_profit_pct <= self.stop_loss_pct:
            issues['risk_reward_ratio'] = False
        
        # Check volume threshold
        if self.volume_threshold < 0.5 or self.volume_threshold > 5.0:
            issues['volume_threshold'] = False
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = self.get_strategy_stats()
        momentum_info = {
            'strategy_type': 'momentum',
            'parameters': {
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'ma_short': self.ma_short_period,
                'ma_long': self.ma_long_period,
                'volume_threshold': self.volume_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'validation': self.validate_parameters()
        }
        
        return {**base_info, **momentum_info}

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'ma_short': 10,
        'ma_long': 20,
        'volume_threshold': 1.2,
        'stop_loss_percentage': 0.05,
        'take_profit_percentage': 0.10,
        'max_position_size': '0.1'
    }
    
    # Create strategy
    strategy = MomentumStrategy(config)
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(100) * 100),
        'volume': np.abs(np.random.randn(100) * 1000) + 1000,
        'rsi': 50 + np.random.randn(100) * 20,
        'sma_10': 50000 + np.cumsum(np.random.randn(100) * 50),
        'sma_20': 50000 + np.cumsum(np.random.randn(100) * 30),
        'volume_ratio': 1 + np.abs(np.random.randn(100) * 0.5)
    }, index=dates)
    
    # Test signal generation
    signals = strategy.generate_signals(test_data)
    print(f"Generated {len(signals)} signals")
    
    # Test momentum analysis
    momentum_analysis = strategy.analyze_momentum_strength(test_data)
    print(f"Momentum analysis: {momentum_analysis}")
    
    # Test parameter validation
    validation = strategy.validate_parameters()
    print(f"Parameter validation: {validation}")
    
    # Get strategy info
    info = strategy.get_strategy_info()
    print(f"Strategy info: {info}")