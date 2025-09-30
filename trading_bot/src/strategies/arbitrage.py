# src/strategies/arbitrage.py
from .base import BaseStrategy, Signal, SignalType, Position
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Tuple

class ArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage and pairs trading strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ArbitrageStrategy", config)
        
        # Strategy parameters
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.zscore_entry = config.get('zscore_entry', 2.0)
        self.zscore_exit = config.get('zscore_exit', 0.5)
        self.lookback_period = config.get('lookback_period', 30)
        self.min_spread_threshold = config.get('min_spread_threshold', 0.001)  # 0.1%
        
        # Risk parameters
        self.stop_loss_pct = config.get('stop_loss_percentage', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_percentage', 0.04)  # 4%
        self.max_position_size = Decimal(str(config.get('max_position_size', '0.05')))
        
        # Pairs tracking
        self.pairs_data = {}  # Store historical data for pairs
        self.correlation_matrix = {}
    
    def update_pairs_data(self, symbol: str, price_data: pd.Series):
        """Update price data for pairs analysis"""
        if symbol not in self.pairs_data:
            self.pairs_data[symbol] = []
        
        self.pairs_data[symbol].append({
            'timestamp': datetime.now(),
            'price': float(price_data['close']),
            'returns': float(price_data.get('price_change', 0))
        })
        
        # Keep only recent data
        if len(self.pairs_data[symbol]) > self.lookback_period * 2:
            self.pairs_data[symbol] = self.pairs_data[symbol][-self.lookback_period * 2:]
    
    def find_cointegrated_pairs(self, symbols: List[str]) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs for arbitrage opportunities"""
        pairs = []
        
        if len(symbols) < 2:
            return pairs
        
        # Calculate correlations between all pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                if (symbol1 in self.pairs_data and symbol2 in self.pairs_data and
                    len(self.pairs_data[symbol1]) >= self.lookback_period and
                    len(self.pairs_data[symbol2]) >= self.lookback_period):
                    
                    # Get recent returns
                    returns1 = [d['returns'] for d in self.pairs_data[symbol1][-self.lookback_period:]]
                    returns2 = [d['returns'] for d in self.pairs_data[symbol2][-self.lookback_period:]]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    if abs(correlation) > self.correlation_threshold:
                        pairs.append((symbol1, symbol2, correlation))
        
        return pairs
    
    def calculate_spread_zscore(self, symbol1: str, symbol2: str) -> Tuple[float, float]:
        """Calculate the z-score of the spread between two assets"""
        if (symbol1 not in self.pairs_data or symbol2 not in self.pairs_data or
            len(self.pairs_data[symbol1]) < self.lookback_period or
            len(self.pairs_data[symbol2]) < self.lookback_period):
            return 0.0, 0.0
        
        # Get recent prices
        prices1 = [d['price'] for d in self.pairs_data[symbol1][-self.lookback_period:]]
        prices2 = [d['price'] for d in self.pairs_data[symbol2][-self.lookback_period:]]
        
        # Calculate price ratio (spread)
        spreads = [p1 / p2 for p1, p2 in zip(prices1, prices2)]
        
        # Calculate z-score
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread == 0:
            return 0.0, 0.0
        
        current_spread = spreads[-1]
        zscore = (current_spread - mean_spread) / std_spread
        
        return zscore, current_spread
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate arbitrage signals based on pairs trading"""
        signals = []
        
        # Update pairs data with current prices
        for symbol in data.index:
            self.update_pairs_data(symbol, data.loc[symbol])
        
        # Find arbitrage opportunities
        symbols = list(data.index)
        cointegrated_pairs = self.find_cointegrated_pairs(symbols)
        
        for symbol1, symbol2, correlation in cointegrated_pairs:
            zscore, current_spread = self.calculate_spread_zscore(symbol1, symbol2)
            
            if abs(zscore) < self.zscore_entry:
                continue
            
            # Get current prices
            price1 = Decimal(str(data.loc[symbol1]['close']))
            price2 = Decimal(str(data.loc[symbol2]['close']))
            
            # Calculate signal strength based on z-score magnitude
            signal_strength = min(1.0, abs(zscore) / 4.0)
            
            # Generate signals for pair
            if zscore > self.zscore_entry:  # symbol1 overpriced relative to symbol2
                # Short symbol1, long symbol2
                signals.extend([
                    Signal(
                        symbol=symbol1,
                        signal_type=SignalType.SELL,
                        strength=signal_strength,
                        price=price1,
                        timestamp=datetime.now(),
                        metadata={
                            'pair_symbol': symbol2,
                            'zscore': zscore,
                            'spread': current_spread,
                            'correlation': correlation,
                            'strategy_type': 'pairs_arbitrage'
                        }
                    ),
                    Signal(
                        symbol=symbol2,
                        signal_type=SignalType.BUY,
                        strength=signal_strength,
                        price=price2,
                        timestamp=datetime.now(),
                        metadata={
                            'pair_symbol': symbol1,
                            'zscore': -zscore,  # Inverse for the pair
                            'spread': 1/current_spread,
                            'correlation': correlation,
                            'strategy_type': 'pairs_arbitrage'
                        }
                    )
                ])
            
            elif zscore < -self.zscore_entry:  # symbol1 underpriced relative to symbol2
                # Long symbol1, short symbol2
                signals.extend([
                    Signal(
                        symbol=symbol1,
                        signal_type=SignalType.BUY,
                        strength=signal_strength,
                        price=price1,
                        timestamp=datetime.now(),
                        metadata={
                            'pair_symbol': symbol2,
                            'zscore': zscore,
                            'spread': current_spread,
                            'correlation': correlation,
                            'strategy_type': 'pairs_arbitrage'
                        }
                    ),
                    Signal(
                        symbol=symbol2,
                        signal_type=SignalType.SELL,
                        strength=signal_strength,
                        price=price2,
                        timestamp=datetime.now(),
                        metadata={
                            'pair_symbol': symbol1,
                            'zscore': -zscore,
                            'spread': 1/current_spread,
                            'correlation': correlation,
                            'strategy_type': 'pairs_arbitrage'
                        }
                    )
                ])
        
        # Add signals to history
        self.signals_history.extend(signals)
        return signals
    
    def should_enter_position(self, signal: Signal, current_positions: Dict[str, Position]) -> bool:
        """Determine if should enter a position based on signal"""
        # For pairs trading, we need both legs
        if signal.metadata and signal.metadata.get('strategy_type') == 'pairs_arbitrage':
            pair_symbol = signal.metadata.get('pair_symbol')
            
            # Check if we already have positions in this pair
            if signal.symbol in current_positions or pair_symbol in current_positions:
                return False
            
            # Only enter on strong signals
            if signal.strength < 0.7:
                return False
            
            return True
        
        return False
    
    def should_exit_position(self, position: Position, current_data: pd.Series) -> bool:
        """Determine if should exit a position"""
        # Standard stop loss and take profit
        if position.pnl_percentage <= -self.stop_loss_pct * 100:
            return True
        
        if position.pnl_percentage >= self.take_profit_pct * 100:
            return True
        
        # Check if spread has mean reverted (for pairs positions)
        # This would need access to the pair data, which could be added
        # to the position metadata in a full implementation
        
        return False

# Example usage and testing
if __name__ == "__main__":
    # Test strategy configurations
    momentum_config = {
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
    
    mean_reversion_config = {
        'bb_period': 20,
        'bb_std': 2,
        'rsi_period': 14,
        'rsi_extreme_oversold': 20,
        'rsi_extreme_overbought': 80,
        'mean_reversion_threshold': 2.0,
        'stop_loss_percentage': 0.03,
        'take_profit_percentage': 0.06,
        'max_position_size': '0.1'
    }
    
    arbitrage_config = {
        'correlation_threshold': 0.8,
        'zscore_entry': 2.0,
        'zscore_exit': 0.5,
        'lookback_period': 30,
        'min_spread_threshold': 0.001,
        'stop_loss_percentage': 0.02,
        'take_profit_percentage': 0.04,
        'max_position_size': '0.05'
    }
'''
    # Initialize strategies
    momentum_strategy = MomentumStrategy(momentum_config)
    mean_reversion_strategy = MeanReversionStrategy(mean_reversion_config)
    arbitrage_strategy = ArbitrageStrategy(arbitrage_config)
    
    print("Strategies initialized successfully!")
    print(f"Momentum strategy: {momentum_strategy.name}")
    print(f"Mean reversion strategy: {mean_reversion_strategy.name}")
    print(f"Arbitrage strategy: {arbitrage_strategy.name}"), Position]) -> bool:
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
'''