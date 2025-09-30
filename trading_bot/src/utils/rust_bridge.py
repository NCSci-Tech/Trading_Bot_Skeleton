# src/utils/rust_bridge.py
"""
Bridge module for integrating Rust components with Python trading bot
"""

try:
    from indicators
    import orderbook  
    import backtesting
    RUST_AVAILABLE = True
    print("Rust modules loaded successfully")
except ImportError as e:
    print(f"Rust modules not available: {e}")
    print("Install Rust modules with: maturin develop")
    RUST_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

class RustIndicators:
    """Wrapper for Rust-based technical indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not RUST_AVAILABLE:
            self.logger.warning("Rust indicators not available, falling back to Python/TA-Lib")
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if RUST_AVAILABLE:
            try:
                return indicators.sma(prices, period)
            except Exception as e:
                self.logger.warning(f"Rust SMA failed, using fallback: {e}")
        
        # Fallback to pandas
        return pd.Series(prices).rolling(period).mean().values
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if RUST_AVAILABLE:
            try:
                return indicators.ema(prices, period)
            except Exception as e:
                self.logger.warning(f"Rust EMA failed, using fallback: {e}")
        
        # Fallback to pandas
        return pd.Series(prices).ewm(span=period).mean().values
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        if RUST_AVAILABLE:
            try:
                return indicators.rsi(prices, period)
            except Exception as e:
                self.logger.warning(f"Rust RSI failed, using fallback: {e}")
        
        # Fallback implementation
        try:
            import talib
            return talib.RSI(prices, timeperiod=period)
        except ImportError:
            # Pure Python fallback
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(period).mean()
            avg_losses = pd.Series(losses).rolling(period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return np.concatenate([[np.nan], rsi.values])
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        if RUST_AVAILABLE:
            try:
                return indicators.macd(prices, fast, slow, signal)
            except Exception as e:
                self.logger.warning(f"Rust MACD failed, using fallback: {e}")
        
        # Fallback implementation
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        if RUST_AVAILABLE:
            try:
                return indicators.bollinger_bands(prices, period, std_dev)
            except Exception as e:
                self.logger.warning(f"Rust Bollinger Bands failed, using fallback: {e}")
        
        # Fallback implementation
        sma = pd.Series(prices).rolling(period).mean()
        std = pd.Series(prices).rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.values, sma.values, lower.values
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        if RUST_AVAILABLE:
            try:
                return indicators.stochastic(high, low, close, k_period, d_period)
            except Exception as e:
                self.logger.warning(f"Rust Stochastic failed, using fallback: {e}")
        
        # Fallback implementation using talib
        try:
            import talib
            return talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        except ImportError:
            # Pure Python fallback (simplified)
            k_values = []
            for i in range(len(close)):
                if i < k_period - 1:
                    k_values.append(np.nan)
                else:
                    window_high = np.max(high[i-k_period+1:i+1])
                    window_low = np.min(low[i-k_period+1:i+1])
                    if window_high != window_low:
                        k = ((close[i] - window_low) / (window_high - window_low)) * 100
                    else:
                        k = 50
                    k_values.append(k)
            
            k_array = np.array(k_values)
            d_values = pd.Series(k_array).rolling(d_period).mean().values
            return k_array, d_values
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if RUST_AVAILABLE:
            try:
                return indicators.atr(high, low, close, period)
            except Exception as e:
                self.logger.warning(f"Rust ATR failed, using fallback: {e}")
        
        # Fallback implementation
        try:
            import talib
            return talib.ATR(high, low, close, timeperiod=period)
        except ImportError:
            # Pure Python fallback
            tr_list = []
            for i in range(len(close)):
                if i == 0:
                    tr = high[i] - low[i]
                else:
                    hl = high[i] - low[i]
                    hc = abs(high[i] - close[i-1])
                    lc = abs(low[i] - close[i-1])
                    tr = max(hl, hc, lc)
                tr_list.append(tr)
            
            return pd.Series(tr_list).rolling(period).mean().values
    
    def batch_calculate(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate multiple indicators in batch (optimized in Rust)"""
        if RUST_AVAILABLE:
            try:
                return indicators.batch_indicators(prices)
            except Exception as e:
                self.logger.warning(f"Rust batch indicators failed, using fallback: {e}")
        
        # Fallback: calculate individually
        results = {}
        
        try:
            results['sma_10'] = self.calculate_sma(prices, 10)
            results['sma_20'] = self.calculate_sma(prices, 20)
            results['sma_50'] = self.calculate_sma(prices, 50)
            results['rsi_14'] = self.calculate_rsi(prices, 14)
            results['ema_12'] = self.calculate_ema(prices, 12)
            results['ema_26'] = self.calculate_ema(prices, 26)
        except Exception as e:
            self.logger.error(f"Fallback batch calculation failed: {e}")
        
        return results
    
    def calculate_momentum(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate price momentum"""
        if RUST_AVAILABLE:
            try:
                return indicators.momentum(prices, period)
            except Exception as e:
                self.logger.warning(f"Rust momentum failed, using fallback: {e}")
        
        # Fallback implementation
        result = np.full_like(prices, np.nan)
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                result[i] = prices[i] / prices[i - period]
        return result
    
    def calculate_roc(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change"""
        if RUST_AVAILABLE:
            try:
                return indicators.roc(prices, period)
            except Exception as e:
                self.logger.warning(f"Rust ROC failed, using fallback: {e}")
        
        # Fallback implementation
        result = np.full_like(prices, np.nan)
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                result[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
        return result

class RustOrderBook:
    """Wrapper for Rust-based order book processing"""
    
    def __init__(self, symbol: str = "UNKNOWN"):
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        if RUST_AVAILABLE:
            try:
                self.orderbook = orderbook.PyOrderBook(symbol)
            except Exception as e:
                self.logger.error(f"Failed to create Rust orderbook: {e}")
                self.orderbook = None
        else:
            self.logger.warning("Rust orderbook not available")
            self.orderbook = None
    
    def update_bid(self, price: float, size: float):
        """Update bid level"""
        if self.orderbook:
            try:
                self.orderbook.update_bid(price, size)
            except Exception as e:
                self.logger.error(f"Error updating bid: {e}")
    
    def update_ask(self, price: float, size: float):
        """Update ask level"""
        if self.orderbook:
            try:
                self.orderbook.update_ask(price, size)
            except Exception as e:
                self.logger.error(f"Error updating ask: {e}")
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (price, size)"""
        if self.orderbook:
            try:
                return self.orderbook.get_best_bid()
            except Exception as e:
                self.logger.error(f"Error getting best bid: {e}")
        return None
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (price, size)"""
        if self.orderbook:
            try:
                return self.orderbook.get_best_ask()
            except Exception as e:
                self.logger.error(f"Error getting best ask: {e}")
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        if self.orderbook:
            try:
                return self.orderbook.get_spread()
            except Exception as e:
                self.logger.error(f"Error getting spread: {e}")
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        if self.orderbook:
            try:
                return self.orderbook.get_mid_price()
            except Exception as e:
                self.logger.error(f"Error getting mid price: {e}")
        return None
    
    def get_depth(self, levels: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get order book depth"""
        if self.orderbook:
            try:
                return self.orderbook.get_depth(levels)
            except Exception as e:
                self.logger.error(f"Error getting depth: {e}")
        return ([], [])
    
    def calculate_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance"""
        if self.orderbook:
            try:
                return self.orderbook.calculate_imbalance(levels)
            except Exception as e:
                self.logger.error(f"Error calculating imbalance: {e}")
        return 0.0
    
    def estimate_market_impact(self, quantity: float, side: str) -> Optional[float]:
        """Estimate market impact of an order"""
        if self.orderbook:
            try:
                return self.orderbook.estimate_market_impact(quantity, side)
            except Exception as e:
                self.logger.error(f"Error estimating market impact: {e}")
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert orderbook to dictionary"""
        if self.orderbook:
            try:
                # This returns a Python dict from Rust
                return self.orderbook.to_dict()
            except Exception as e:
                self.logger.error(f"Error converting to dict: {e}")
        return {}
    
    def calculate_vwap(self, prices: List[float], volumes: List[float]) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        if RUST_AVAILABLE:
            try:
                return orderbook.calculate_vwap(prices, volumes)
            except Exception as e:
                self.logger.error(f"Error calculating VWAP: {e}")
        
        # Fallback implementation
        if len(prices) != len(volumes) or not prices:
            return None
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return None
        
        weighted_sum = sum(p * v for p, v in zip(prices, volumes))
        return weighted_sum / total_volume
    
    def calculate_twap(self, prices: List[float], timestamps: List[int]) -> Optional[float]:
        """Calculate Time Weighted Average Price"""
        if RUST_AVAILABLE:
            try:
                return orderbook.calculate_twap(prices, timestamps)
            except Exception as e:
                self.logger.error(f"Error calculating TWAP: {e}")
        
        # Fallback implementation
        if len(prices) != len(timestamps) or len(prices) < 2:
            return None
        
        weighted_sum = 0.0
        total_time = 0
        
        for i in range(1, len(prices)):
            time_diff = timestamps[i] - timestamps[i-1]
            weighted_sum += prices[i-1] * time_diff
            total_time += time_diff
        
        return weighted_sum / total_time if total_time > 0 else None

class RustBacktester:
    """Wrapper for Rust-based backtesting engine"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001, slippage: float = 0.0001):
        self.logger = logging.getLogger(__name__)
        if RUST_AVAILABLE:
            try:
                self.engine = backtesting.BacktestEngine(initial_capital, commission_rate, slippage)
                self.logger.info("Rust backtesting engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to create Rust backtest engine: {e}")
                self.engine = None
        else:
            self.logger.warning("Rust backtesting engine not available")
            self.engine = None
    
    def execute_trade(self, timestamp: str, symbol: str, side: str, price: float, quantity: float) -> bool:
        """Execute a trade in the backtest"""
        if self.engine:
            try:
                self.engine.execute_trade(timestamp, symbol, side, price, quantity)
                return True
            except Exception as e:
                self.logger.error(f"Error executing trade: {e}")
        return False
    
    def update_prices(self, prices: Dict[str, float], timestamp: str) -> bool:
        """Update current prices"""
        if self.engine:
            try:
                self.engine.update_prices(prices, timestamp)
                return True
            except Exception as e:
                self.logger.error(f"Error updating prices: {e}")
        return False
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get backtest results"""
        if self.engine:
            try:
                results = self.engine.get_results()
                # Convert Rust results to Python dict
                return {
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'win_rate': results.win_rate,
                    'profit_factor': results.profit_factor,
                    'total_trades': results.total_trades,
                    'winning_trades': results.winning_trades,
                    'losing_trades': results.losing_trades,
                    'avg_win': results.avg_win,
                    'avg_loss': results.avg_loss,
                    'avg_trade_duration': results.avg_trade_duration,
                    'max_consecutive_wins': results.max_consecutive_wins,
                    'max_consecutive_losses': results.max_consecutive_losses,
                    'calmar_ratio': results.calmar_ratio,
                    'sortino_ratio': results.sortino_ratio,
                    'equity_curve': results.equity_curve,
                    'drawdown_curve': results.drawdown_curve,
                    'returns': results.returns,
                    'trade_pnls': results.trade_pnls
                }
            except Exception as e:
                self.logger.error(f"Error getting backtest results: {e}")
        return None
    
    def reset(self) -> bool:
        """Reset the backtest engine"""
        if self.engine:
            try:
                self.engine.reset()
                return True
            except Exception as e:
                self.logger.error(f"Error resetting engine: {e}")
        return False
    
    def get_current_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get current positions (symbol -> (quantity, avg_price, unrealized_pnl))"""
        if self.engine:
            try:
                return self.engine.get_current_positions()
            except Exception as e:
                self.logger.error(f"Error getting positions: {e}")
        return {}
    
    def get_trade_history(self) -> List[Tuple[str, str, str, float, float, float, str]]:
        """Get trade history"""
        if self.engine:
            try:
                return self.engine.get_trade_history()
            except Exception as e:
                self.logger.error(f"Error getting trade history: {e}")
        return []

def run_parallel_backtest(initial_capital: float, parameter_sets: List[Dict[str, float]], 
                         trades_data: List[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
    """Run parallel backtests with different parameter sets"""
    if not RUST_AVAILABLE:
        return None
    
    try:
        results = backtesting.parallel_backtest(initial_capital, parameter_sets, trades_data)
        return [
            {
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'total_trades': r.total_trades,
                'equity_curve': r.equity_curve
            }
            for r in results
        ]
    except Exception as e:
        logging.getLogger(__name__).error(f"Parallel backtest failed: {e}")
        return None

# Performance benchmark
def benchmark_indicators(prices: np.ndarray, iterations: int = 100) -> Dict[str, float]:
    """Benchmark Rust vs Python indicator performance"""
    import time
    
    results = {}
    rust_indicators = RustIndicators()
    
    # Benchmark SMA
    start_time = time.time()
    for _ in range(iterations):
        rust_indicators.calculate_sma(prices, 20)
    rust_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(iterations):
        pd.Series(prices).rolling(20).mean().values
    python_time = time.time() - start_time
    
    results['sma_speedup'] = python_time / rust_time if rust_time > 0 else 0
    
    # Benchmark RSI
    start_time = time.time()
    for _ in range(iterations):
        rust_indicators.calculate_rsi(prices, 14)
    rust_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(iterations):
        # Simplified Python RSI for benchmark
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        pd.Series(gains).rolling(14).mean()
    python_time = time.time() - start_time
    
    results['rsi_speedup'] = python_time / rust_time if rust_time > 0 else 0
    
    return results

# Example usage and testing
if __name__ == "__main__":
    print("Testing Rust-Python bridge...")
    
    # Test indicators
    print("\nTesting technical indicators...")
    rust_indicators = RustIndicators()
    
    # Generate test data
    np.random.seed(42)
    test_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    sma_result = rust_indicators.calculate_sma(test_prices, 20)
    print(f"SMA calculated: {len(sma_result)} points")
    
    rsi_result = rust_indicators.calculate_rsi(test_prices)
    print(f"RSI calculated: {len(rsi_result)} points")
    
    batch_result = rust_indicators.batch_calculate(test_prices)
    print(f"Batch indicators: {list(batch_result.keys())}")
    
    # Test order book
    print("\nTesting order book...")
    ob = RustOrderBook("BTCUSD")
    ob.update_bid(50000, 1.5)
    ob.update_ask(50100, 2.0)
    
    spread = ob.get_spread()
    print(f"Spread: {spread}")
    
    # Test backtester
    print("\nTesting backtester...")
    bt = RustBacktester(10000, 0.001, 0.0001)
    
    if bt.execute_trade("2023-01-01T00:00:00Z", "BTCUSD", "buy", 50000, 0.1):
        print("Trade executed successfully")
    
    # Performance benchmark
    if RUST_AVAILABLE:
        print("\n⚡ Performance benchmark...")
        benchmark_results = benchmark_indicators(test_prices, iterations=10)
        for metric, speedup in benchmark_results.items():
            print(f"{metric}: {speedup:.2f}x speedup")
    
    print("\nRust-Python bridge testing complete!")