# src/risk/manager.py
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np
from ..strategies.base import Signal, Position, SignalType

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_portfolio_risk: float  # Maximum portfolio risk per trade
    max_daily_loss: float      # Maximum daily loss allowed
    max_drawdown: float        # Maximum drawdown from peak
    max_positions: int         # Maximum number of open positions
    max_position_size: Decimal # Maximum position size
    max_leverage: float        # Maximum leverage allowed

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: Decimal
    available_balance: Decimal
    total_pnl: Decimal
    daily_pnl: Decimal
    max_drawdown: float
    peak_value: Decimal
    open_positions: int
    total_exposure: Decimal

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.limits = RiskLimits(
            max_portfolio_risk=config.risk.max_portfolio_risk,
            max_daily_loss=config.risk.max_daily_loss,
            max_drawdown=config.risk.max_drawdown,
            max_positions=getattr(config.risk, 'max_positions', 10),
            max_position_size=Decimal(str(getattr(config.risk, 'max_position_size', '1000'))),
            max_leverage=getattr(config.risk, 'max_leverage', 1.0)
        )
        
        # Portfolio tracking
        self.initial_balance = Decimal('0')
        self.daily_start_balance = Decimal('0')
        self.peak_balance = Decimal('0')
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Risk state
        self.emergency_stop = False
        self.daily_loss_limit_reached = False
        self.drawdown_limit_reached = False
        
        # Historical tracking
        self.portfolio_history = []
        self.drawdown_history = []
        self.daily_returns = []
        
    def set_initial_balance(self, balance: Decimal):
        """Set initial portfolio balance"""
        self.initial_balance = balance
        self.daily_start_balance = balance
        self.peak_balance = balance
        self.logger.info(f"Initial balance set to {balance}")
    
    def update_portfolio_metrics(self, current_balance: Decimal, 
                                positions: Dict[str, Position]) -> PortfolioMetrics:
        """Update and return current portfolio metrics"""
        # Reset daily tracking if new day
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self._reset_daily_limits(current_balance)
        
        # Calculate metrics
        total_pnl = sum(pos.pnl for pos in positions.values())
        daily_pnl = current_balance - self.daily_start_balance
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        current_drawdown = float((self.peak_balance - current_balance) / self.peak_balance) if self.peak_balance > 0 else 0.0
        
        # Calculate exposure
        total_exposure = sum(pos.entry_price * pos.quantity for pos in positions.values())
        
        # Store historical data
        self.portfolio_history.append({
            'timestamp': now,
            'balance': current_balance,
            'pnl': total_pnl,
            'drawdown': current_drawdown
        })
        
        # Keep only last 1000 records
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        metrics = PortfolioMetrics(
            total_value=current_balance,
            available_balance=current_balance - total_exposure,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=current_drawdown,
            peak_value=self.peak_balance,
            open_positions=len(positions),
            total_exposure=total_exposure
        )
        
        # Check risk limits
        self._check_risk_limits(metrics)
        
        return metrics
    
    def _reset_daily_limits(self, current_balance: Decimal):
        """Reset daily risk tracking"""
        self.daily_start_balance = current_balance
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_loss_limit_reached = False
        
        # Calculate daily return
        if len(self.portfolio_history) > 0:
            prev_balance = self.portfolio_history[-1]['balance']
            daily_return = float((current_balance - prev_balance) / prev_balance)
            self.daily_returns.append(daily_return)
            
            # Keep only last 252 days (trading year)
            if len(self.daily_returns) > 252:
                self.daily_returns = self.daily_returns[-252:]
        
        self.logger.info("Daily risk limits reset")
    
    def _check_risk_limits(self, metrics: PortfolioMetrics):
        """Check if any risk limits are breached"""
        # Daily loss limit
        if self.daily_start_balance > 0:
            daily_loss_pct = float(abs(metrics.daily_pnl) / self.daily_start_balance)
            if daily_loss_pct > self.limits.max_daily_loss and metrics.daily_pnl < 0:
                if not self.daily_loss_limit_reached:
                    self.daily_loss_limit_reached = True
                    self.logger.critical(f"Daily loss limit reached: {daily_loss_pct:.2%}")
        
        # Maximum drawdown limit
        if metrics.max_drawdown > self.limits.max_drawdown:
            if not self.drawdown_limit_reached:
                self.drawdown_limit_reached = True
                self.logger.critical(f"Maximum drawdown limit reached: {metrics.max_drawdown:.2%}")
        
        # Emergency stop conditions
        emergency_loss_threshold = 0.15  # 15% emergency stop
        daily_loss_pct = float(abs(metrics.daily_pnl) / self.daily_start_balance) if self.daily_start_balance > 0 else 0
        
        if (daily_loss_pct > emergency_loss_threshold or 
            metrics.max_drawdown > emergency_loss_threshold):
            if not self.emergency_stop:
                self.emergency_stop = True
                self.logger.critical("EMERGENCY STOP ACTIVATED!")
    
    def can_open_position(self, signal: Signal, current_positions: Dict[str, Position], 
                         current_balance: Decimal) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        # Emergency stop check
        if self.emergency_stop:
            return False, "Emergency stop active"
        
        # Daily loss limit check
        if self.daily_loss_limit_reached:
            return False, "Daily loss limit reached"
        
        # Drawdown limit check
        if self.drawdown_limit_reached:
            return False, "Drawdown limit reached"
        
        # Maximum positions check
        if len(current_positions) >= self.limits.max_positions:
            return False, "Maximum positions limit reached"
        
        # Check if we have enough balance
        position_size = self.calculate_position_size(signal, current_balance, current_positions)
        required_margin = position_size * signal.price
        
        if required_margin > current_balance * Decimal('0.9'):  # Keep 10% buffer
            return False, "Insufficient balance"
        
        # Check correlation limits (avoid too many correlated positions)
        correlation_limit = 0.7
        if self._check_correlation_limit(signal, current_positions, correlation_limit):
            return False, "Too many correlated positions"
        
        return True, "Position can be opened"
    
    def calculate_position_size(self, signal: Signal, current_balance: Decimal, 
                              current_positions: Dict[str, Position]) -> Decimal:
        """Calculate appropriate position size based on risk management"""
        # Risk per trade (percentage of portfolio)
        risk_per_trade = Decimal(str(self.limits.max_portfolio_risk))
        risk_amount = current_balance * risk_per_trade
        
        # Estimate stop loss distance (could be improved with volatility measures)
        estimated_stop_loss = Decimal('0.05')  # 5% default stop loss
        if signal.metadata:
            # Use strategy-specific stop loss if available
            estimated_stop_loss = Decimal(str(signal.metadata.get('estimated_stop_loss', 0.05)))
        
        # Position size based on risk
        position_size = risk_amount / (signal.price * estimated_stop_loss)
        
        # Apply maximum position size limit
        position_size = min(position_size, self.limits.max_position_size)
        
        # Apply signal strength scaling
        position_size = position_size * Decimal(str(signal.strength))
        
        # Apply portfolio heat adjustment (reduce size if too many positions)
        portfolio_heat = len(current_positions) / self.limits.max_positions
        if portfolio_heat > 0.7:  # If more than 70% of max positions
            position_size = position_size * Decimal(str(1 - portfolio_heat * 0.3))
        
        # Ensure minimum viable position size
        min_position_size = Decimal('0.001')  # Minimum 0.001 units
        position_size = max(position_size, min_position_size)
        
        # Apply volatility adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(signal)
        position_size = position_size * Decimal(str(volatility_adjustment))
        
        self.logger.debug(f"Calculated position size for {signal.symbol}: {position_size}")
        return position_size
    
    def _calculate_volatility_adjustment(self, signal: Signal) -> float:
        """Calculate position size adjustment based on volatility"""
        # Default adjustment if no volatility data
        if not signal.metadata or 'volatility' not in signal.metadata:
            return 1.0
        
        volatility = signal.metadata['volatility']
        
        # Reduce position size for high volatility assets
        if volatility > 0.05:  # 5% volatility threshold
            return max(0.5, 1 - (volatility - 0.05) * 2)
        else:
            return 1.0
    
    def _check_correlation_limit(self, signal: Signal, current_positions: Dict[str, Position], 
                               limit: float) -> bool:
        """Check if opening position would exceed correlation limits"""
        # Simplified correlation check based on asset class
        # In practice, you'd calculate actual price correlations
        
        signal_asset_class = self._get_asset_class(signal.symbol)
        similar_positions = 0
        
        for symbol, position in current_positions.items():
            if self._get_asset_class(symbol) == signal_asset_class:
                similar_positions += 1
        
        # Limit similar positions to 50% of max positions
        max_similar = max(1, int(self.limits.max_positions * 0.5))
        return similar_positions >= max_similar
    
    def _get_asset_class(self, symbol: str) -> str:
        """Get asset class for correlation analysis"""
        # Simplified asset class mapping
        if 'BTC' in symbol or 'XBT' in symbol:
            return 'crypto_major'
        elif 'ETH' in symbol:
            return 'crypto_major'
        elif any(x in symbol for x in ['USD', 'EUR', 'GBP', 'JPY']):
            return 'forex'
        else:
            return 'crypto_alt'
    
    def should_reduce_exposure(self, metrics: PortfolioMetrics) -> bool:
        """Check if we should reduce overall exposure"""
        exposure_ratio = float(metrics.total_exposure / metrics.total_value) if metrics.total_value > 0 else 0
        max_exposure_ratio = 0.8  # Maximum 80% exposure
        
        return (exposure_ratio > max_exposure_ratio or 
                metrics.max_drawdown > self.limits.max_drawdown * 0.8)  # 80% of drawdown limit
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate advanced portfolio performance metrics"""
        if len(self.daily_returns) < 10:
            return {'insufficient_data': True}
        
        returns_array = np.array(self.daily_returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns_array) - 1
        annual_return = (1 + total_return) ** (252 / len(returns_array)) - 1
        annual_volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Win rate
        winning_days = np.sum(returns_array > 0)
        win_rate = winning_days / len(returns_array)
        
        # Average win/loss
        winning_returns = returns_array[returns_array > 0]
        losing_returns = returns_array[returns_array < 0]
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        
        # Profit factor
        profit_factor = abs(np.sum(winning_returns) / np.sum(losing_returns)) if len(losing_returns) > 0 else float('inf')
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'total_trades': len(returns_array)
        }
    
    def get_risk_status(self) -> Dict[str, any]:
        """Get current risk management status"""
        return {
            'emergency_stop': self.emergency_stop,
            'daily_loss_limit_reached': self.daily_loss_limit_reached,
            'drawdown_limit_reached': self.drawdown_limit_reached,
            'limits': {
                'max_portfolio_risk': self.limits.max_portfolio_risk,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_drawdown': self.limits.max_drawdown,
                'max_positions': self.limits.max_positions,
                'max_position_size': float(self.limits.max_position_size),
                'max_leverage': self.limits.max_leverage
            },
            'current_state': {
                'initial_balance': float(self.initial_balance),
                'daily_start_balance': float(self.daily_start_balance),
                'peak_balance': float(self.peak_balance),
                'portfolio_records': len(self.portfolio_history),
                'daily_returns_count': len(self.daily_returns)
            }
        }
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual override)"""
        self.emergency_stop = False
        self.logger.warning("Emergency stop manually reset")
    
    def adjust_risk_limits(self, new_limits: Dict[str, float]):
        """Adjust risk management limits"""
        for key, value in new_limits.items():
            if hasattr(self.limits, key):
                setattr(self.limits, key, value)
                self.logger.info(f"Risk limit {key} updated to {value}")
    
    def simulate_position_impact(self, signal: Signal, current_balance: Decimal) -> Dict[str, float]:
        """Simulate the impact of opening a position"""
        position_size = self.calculate_position_size(signal, current_balance, {})
        position_value = position_size * signal.price
        
        # Estimate potential profit/loss scenarios
        scenarios = {
            'stop_loss': -0.05,  # 5% loss
            'take_profit': 0.10,  # 10% profit
            'breakeven': 0.0
        }
        
        impact_analysis = {}
        for scenario, price_change in scenarios.items():
            pnl = position_value * Decimal(str(price_change))
            portfolio_impact = float(pnl / current_balance)
            impact_analysis[scenario] = {
                'pnl': float(pnl),
                'portfolio_impact_pct': portfolio_impact * 100
            }
        
        return {
            'position_size': float(position_size),
            'position_value': float(position_value),
            'portfolio_allocation_pct': float(position_value / current_balance * 100),
            'scenarios': impact_analysis
        }

# Example usage
if __name__ == "__main__":
    from types import SimpleNamespace
    
    # Mock config
    risk_config = SimpleNamespace()
    risk_config.max_portfolio_risk = 0.02
    risk_config.max_daily_loss = 0.05
    risk_config.max_drawdown = 0.10
    
    config = SimpleNamespace()
    config.risk = risk_config
    
    # Test risk manager
    risk_manager = RiskManager(config)
    risk_manager.set_initial_balance(Decimal('10000'))
    
    # Test signal
    signal = Signal(
        symbol="BTCUSD",
        signal_type=SignalType.BUY,
        strength=0.8,
        price=Decimal('50000'),
        timestamp=datetime.now()
    )
    
    # Test position sizing
    position_size = risk_manager.calculate_position_size(signal, Decimal('10000'), {})
    print(f"Calculated position size: {position_size}")
    
    # Test risk status
    status = risk_manager.get_risk_status()
    print(f"Risk status: {status}")