# src/strategies/mean_reversion.py
from .base import BaseStrategy, Signal, SignalType, Position
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MeanReversionStrategy", config)
        
        # Strategy parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_extreme_oversold = config.get('rsi_extreme_oversold', 20)
        self.rsi_extreme_overbought = config.get('rsi_extreme_overbought', 80)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 2.0)
        
        # Risk parameters
        self.stop_loss_pct = config.get('stop_loss_percentage', 0.03)  # 3% for mean reversion
        self.take_profit_pct = config.get('take_profit_percentage', 0.06)  # 6%
        self.max_position_size = Decimal(str(config.get('max_position_size', '0.1')))
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion signals"""
        if len(data) < max(self.bb_period, self.rsi_period):
            return []
        
        signals = []
        latest = data.iloc[-1]
        symbol = latest.name if hasattr(latest, 'name') else 'UNKNOWN'
        
        # Get indicators
        close_price = latest['close']
        bb_upper = latest.get('bb_upper')
        bb_lower = latest.get('bb_lower')
        bb_middle = latest.get('bb_middle')
        bb_position = latest.get('bb_position')  # Position within bands (0 to 1)
        rsi = latest.get('rsi')
        mean_reversion_10 = latest.get('mean_reversion_10')  # From processor
        
        if (pd.isna(bb_upper) or pd.isna(bb_lower) or 
            pd.isna(bb_position) or pd.isna(rsi)):
            return signals
        
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        
        # Oversold condition - price near lower band with low RSI
        if (bb_position < 0.1 and  # Very close to lower band
            rsi < self.rsi_extreme_oversold and
            close_price < bb_lower):
            
            signal_type = SignalType.BUY
            signal_strength = min(1.0, 
                                (0.1 - bb_position) * 2 +  # Band position component
                                (self.rsi_extreme_oversold - rsi) / 20 +  # RSI component
                                abs(mean_reversion_10) / 10 if pd.notna(mean_reversion_10) else 0)
        
        # Overbought condition - price near upper band with high RSI
        elif (bb_position > 0.9 and  # Very close to upper band
              rsi > self.rsi_extreme_overbought and
              close_price > bb_upper):
            
            signal_type = SignalType.SELL
            signal_strength = min(1.0,
                                (bb_position - 0.9) * 2 +  # Band position component
                                (rsi - self.rsi_extreme_overbought) / 20 +  # RSI component
                                abs(mean_reversion_10) / 10 if pd.notna(mean_reversion_10) else 0)
        
        # Additional mean reversion signal based on extreme deviations
        elif pd.notna(mean_reversion_10) and abs(mean_reversion_10) > self.mean_reversion_threshold:
            if mean_reversion_10 < -self.mean_reversion_threshold:  # Extremely undervalued
                signal_type = SignalType.BUY
                signal_strength = min(0.8, abs(mean_reversion_10) / 5)
            elif mean_reversion_10 > self.mean_reversion_threshold:  # Extremely overvalued
                signal_type = SignalType.SELL
                signal_strength = min(0.8, abs(mean_reversion_10) / 5)
        
        # Create signal if strength is sufficient
        if signal_strength > 0.4:  # Higher threshold for mean reversion
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=Decimal(str(close_price)),
                timestamp=datetime.now(),
                metadata={
                    'bb_position': bb_position,
                    'rsi': rsi,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'mean_reversion_10': mean_reversion_10
                }
            )
            signals.append(signal)
            self.signals_history.append(signal)
        
        return signals
    
    def should_enter_position(self, signal: Signal, current_positions: Dict[str, Position]) -> bool:
        """Determine if should enter a position based on signal"""
        # Don't enter if already have position in this symbol
        if signal.symbol in current_positions:
            return False
        
        # Only enter on strong signals (higher threshold for mean reversion)
        if signal.strength < 0.6:
            return False
        
        # Only enter buy/sell signals
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            return True
        
        return False
    
    def should_exit_position(self, position: Position, current_data: pd.Series) -> bool:
        """Determine if should exit a position"""
        # Stop loss and take profit
        if position.pnl_percentage <= -self.stop_loss_pct * 100:
            self.logger.info(f"Stop loss triggered for {position.symbol} at {position.pnl_percentage:.2f}%")
            return True
        
        if position.pnl_percentage >= self.take_profit_pct * 100:
            self.logger.info(f"Take profit triggered for {position.symbol} at {position.pnl_percentage:.2f}%")
            return True
        
        # Mean reversion exit conditions
        bb_position = current_data.get('bb_position')
        rsi = current_data.get('rsi')
        
        if pd.notna(bb_position) and pd.notna(rsi):
            # Exit long position when price moves back to middle of bands
            if position.side == 'long' and bb_position > 0.5 and rsi > 50:
                return True
            # Exit short position when price moves back to middle of bands
            elif position.side == 'short' and bb_position < 0.5 and rsi < 50:
                return True
        
        return False