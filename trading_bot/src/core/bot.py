# src/core/bot.py
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from pathlib import Path

from .config import Config
from ..exchanges.kraken import KrakenExchange
from ..data.collector import DataCollector
from ..data.processor import DataProcessor
from ..strategies.momentum import MomentumStrategy
from ..strategies.base import BaseStrategy, Signal, Position, SignalType
from ..risk.manager import RiskManager

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.logger = self._setup_logging()
        
        # Components
        self.exchanges = {}
        self.data_collector = None
        self.data_processor = None
        self.risk_manager = None
        self.strategies = {}
        
        # State
        self.is_running = False
        self.positions = {}
        self.balance = Decimal('0')
        self.last_update_time = datetime.now()
        self.performance_stats = {}
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # Initialize exchanges
            for exchange_name, exchange_config in self.config.exchanges.items():
                if exchange_name == "kraken":
                    self.exchanges[exchange_name] = KrakenExchange(
                        exchange_config.api_key,
                        exchange_config.api_secret,
                        exchange_config.sandbox
                    )
                    self.logger.info(f"Initialized {exchange_name} exchange")
            
            # Initialize data components
            if self.exchanges:
                primary_exchange = list(self.exchanges.values())[0]
                self.data_collector = DataCollector(self.config, primary_exchange)
                self.data_processor = DataProcessor(self.config)
                self.logger.info("Initialized data collection and processing components")
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)
            self.logger.info("Initialized risk manager")
            
            # Initialize strategies
            for strategy_name, strategy_config in self.config.strategies.items():
                strategy_params = {
                    'enabled': strategy_config.enabled,
                    'symbols': strategy_config.symbols,
                    'max_position_size': str(strategy_config.max_position_size),
                    'stop_loss_percentage': strategy_config.stop_loss_percentage,
                    'take_profit_percentage': strategy_config.take_profit_percentage,
                    **strategy_config.parameters
                }
                
                if 'momentum' in strategy_name.lower():
                    self.strategies[strategy_name] = MomentumStrategy(strategy_params)
                    self.logger.info(f"Initialized momentum strategy: {strategy_name}")
                # Add other strategy types here as needed
            
            self.logger.info("All bot components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
        
        self.logger.info("Starting trading bot...")
        
        try:
            # Connect to exchanges
            await self._connect_exchanges()
            
            # Initialize portfolio
            await self._initialize_portfolio()
            
            # Validate configuration
            if not self.config.validate_config():
                raise Exception("Configuration validation failed")
            
            self.is_running = True
            self.logger.info("Trading bot started successfully")
            
            # Start main trading loop
            await self._run_trading_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping trading bot...")
        self.is_running = False
        
        try:
            # Stop data collection
            if self.data_collector:
                await self.data_collector.stop_collection()
            
            # Optionally close all positions (manual decision in practice)
            if self._should_close_positions_on_stop():
                await self._close_all_positions()
            
            # Disconnect from exchanges
            await self._disconnect_exchanges()
            
            # Save final performance report
            await self._save_performance_report()
            
            self.logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    async def _connect_exchanges(self):
        """Connect to all configured exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                connected = await exchange.connect()
                if connected:
                    self.logger.info(f"Connected to {exchange_name}")
                else:
                    raise Exception(f"Failed to connect to {exchange_name}")
            except Exception as e:
                self.logger.error(f"Connection failed for {exchange_name}: {e}")
                raise
    
    async def _disconnect_exchanges(self):
        """Disconnect from all exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.logger.info(f"Disconnected from {exchange_name}")
            except Exception as e:
                self.logger.warning(f"Error disconnecting from {exchange_name}: {e}")
    
    async def _initialize_portfolio(self):
        """Initialize portfolio balance and positions"""
        try:
            # Get balance from primary exchange
            primary_exchange = list(self.exchanges.values())[0]
            balance_dict = await primary_exchange.get_balance()
            
            # Calculate total balance in base currency (simplified)
            self.balance = sum(balance_dict.values())
            
            # Initialize risk manager with starting balance
            self.risk_manager.set_initial_balance(self.balance)
            
            # Load existing positions if any
            await self._load_existing_positions()
            
            self.logger.info(f"Portfolio initialized - Balance: {self.balance}, Positions: {len(self.positions)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing portfolio: {e}")
            raise
    
    async def _load_existing_positions(self):
        """Load existing open positions from exchange"""
        try:
            primary_exchange = list(self.exchanges.values())[0]
            open_orders = await primary_exchange.get_open_orders()
            
            # Convert open orders to positions (simplified)
            # In practice, you'd have more sophisticated position tracking
            
            self.logger.info(f"Loaded {len(self.positions)} existing positions")
            
        except Exception as e:
            self.logger.warning(f"Error loading existing positions: {e}")
    
    async def _run_trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting main trading loop")
        
        # Get symbols to trade
        all_symbols = set()
        for strategy_config in self.config.strategies.values():
            if strategy_config.enabled:
                all_symbols.update(strategy_config.symbols)
        
        symbols = list(all_symbols)
        self.logger.info(f"Trading symbols: {symbols}")
        
        # Start data collection
        data_collection_task = None
        if symbols:
            data_collection_task = asyncio.create_task(
                self.data_collector.start_collection(symbols, ["1m"])
            )
        
        try:
            # Main trading loop
            while self.is_running:
                loop_start_time = datetime.now()
                
                try:
                    await self._trading_cycle()
                    
                    # Sleep until next cycle (60 seconds - execution time)
                    execution_time = (datetime.now() - loop_start_time).total_seconds()
                    sleep_time = max(1, 60 - execution_time)
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
                
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {e}")
        finally:
            if data_collection_task:
                data_collection_task.cancel()
                try:
                    await data_collection_task
                except asyncio.CancelledError:
                    pass
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        cycle_start_time = datetime.now()
        
        try:
            # Update portfolio metrics
            await self._update_portfolio()
            
            # Generate signals from all strategies
            signals = await self._generate_signals()
            
            # Execute trades based on signals
            if signals:
                await self._execute_trades(signals)
            
            # Update existing positions
            await self._update_positions()
            
            # Check for position exits
            await self._check_position_exits()
            
            # Update performance statistics
            self._update_performance_stats()
            
            # Log cycle summary
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            self.logger.debug(f"Trading cycle completed in {cycle_duration:.2f}s - "
                            f"Signals: {len(signals)}, Positions: {len(self.positions)}")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            raise
    
    async def _update_portfolio(self):
        """Update current portfolio balance and metrics"""
        try:
            primary_exchange = list(self.exchanges.values())[0]
            balance_dict = await primary_exchange.get_balance()
            
            # Update balance
            new_balance = sum(balance_dict.values())
            if abs(new_balance - self.balance) > Decimal('0.01'):  # Only log significant changes
                self.logger.debug(f"Balance updated: {self.balance} -> {new_balance}")
                self.balance = new_balance
            
            # Update risk manager metrics
            metrics = self.risk_manager.update_portfolio_metrics(self.balance, self.positions)
            
            # Log critical risk events
            if self.risk_manager.emergency_stop:
                self.logger.critical("EMERGENCY STOP ACTIVE - All trading halted")
                # Could implement automatic position closure here
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def _generate_signals(self) -> List[Signal]:
        """Generate signals from all active strategies"""
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.is_active:
                continue
            
            try:
                # Get strategy configuration
                strategy_config = self.config.get_strategy_config(strategy_name)
                if not strategy_config or not strategy_config.enabled:
                    continue
                
                # Process each symbol for this strategy
                for symbol in strategy_config.symbols:
                    try:
                        # Get recent market data
                        df = self.data_collector.get_ohlcv_data(symbol, limit=100)
                        
                        if not df.empty and len(df) >= 50:  # Minimum data requirement
                            # Process data (add technical indicators)
                            df_processed = self.data_processor.calculate_technical_indicators(df)
                            df_processed = self.data_processor.calculate_custom_features(df_processed)
                            
                            # Generate signals
                            signals = strategy.generate_signals(df_processed)
                            
                            # Validate and add signals
                            for signal in signals:
                                if strategy.validate_signal(signal):
                                    signal.symbol = symbol  # Ensure symbol is set correctly
                                    all_signals.append(signal)
                                    
                        else:
                            self.logger.debug(f"Insufficient data for {symbol} in {strategy_name}")
                    
                    except Exception as e:
                        self.logger.warning(f"Error generating signals for {symbol} in {strategy_name}: {e}")
            
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {e}")
        
        # Filter and rank signals
        filtered_signals = self._filter_and_rank_signals(all_signals)
        
        if filtered_signals:
            self.logger.info(f"Generated {len(filtered_signals)} signals from {len(all_signals)} total")
        
        return filtered_signals
    
    def _filter_and_rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter and rank signals by strength and risk criteria"""
        if not signals:
            return []
        
        # Remove duplicate signals for same symbol (keep strongest)
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals or signal.strength > symbol_signals[signal.symbol].strength:
                symbol_signals[signal.symbol] = signal
        
        filtered_signals = list(symbol_signals.values())
        
        # Remove signals for symbols we already have positions in
        # (unless it's a close signal)
        final_signals = []
        for signal in filtered_signals:
            if (signal.symbol not in self.positions or 
                signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]):
                final_signals.append(signal)
        
        # Sort by signal strength (highest first)
        final_signals.sort(key=lambda s: s.strength, reverse=True)
        
        # Limit number of signals to process (risk management)
        max_signals = min(5, self.risk_manager.limits.max_positions - len(self.positions))
        return final_signals[:max_signals]
    
    async def _execute_trades(self, signals: List[Signal]):
        """Execute trades based on signals"""
        primary_exchange = list(self.exchanges.values())[0]
        
        for signal in signals:
            try:
                # Check if we can open this position
                can_open, reason = self.risk_manager.can_open_position(signal, self.positions, self.balance)
                if not can_open:
                    self.logger.info(f"Cannot open position for {signal.symbol}: {reason}")
                    continue
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    signal, self.balance, self.positions
                )
                
                if position_size < Decimal('0.001'):  # Minimum viable size
                    self.logger.warning(f"Position size too small for {signal.symbol}: {position_size}")
                    continue
                
                # Execute trade based on signal type
                await self._execute_trade(signal, position_size, primary_exchange)
                
            except Exception as e:
                self.logger.error(f"Error executing trade for {signal.symbol}: {e}")
    
    async def _execute_trade(self, signal: Signal, position_size: Decimal, exchange):
        """Execute a single trade"""
        try:
            if signal.signal_type == SignalType.BUY:
                # Place buy order
                result = await exchange.place_order(
                    symbol=signal.symbol,
                    side="buy",
                    amount=position_size,
                    order_type="market"
                )
                
                if result and 'txid' in result:
                    # Add position to tracking
                    self.positions[signal.symbol] = Position(
                        symbol=signal.symbol,
                        side='long',
                        entry_price=signal.price,
                        current_price=signal.price,
                        quantity=position_size,
                        entry_time=datetime.now(),
                        pnl=Decimal('0'),
                        pnl_percentage=0.0
                    )
                    
                    self.logger.info(f"Opened LONG position: {signal.symbol} {position_size} @ {signal.price}")
            
            elif signal.signal_type == SignalType.SELL:
                # Place sell order (short position)
                result = await exchange.place_order(
                    symbol=signal.symbol,
                    side="sell",
                    amount=position_size,
                    order_type="market"
                )
                
                if result and 'txid' in result:
                    # Add position to tracking
                    self.positions[signal.symbol] = Position(
                        symbol=signal.symbol,
                        side='short',
                        entry_price=signal.price,
                        current_price=signal.price,
                        quantity=position_size,
                        entry_time=datetime.now(),
                        pnl=Decimal('0'),
                        pnl_percentage=0.0
                    )
                    
                    self.logger.info(f"Opened SHORT position: {signal.symbol} {position_size} @ {signal.price}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute {signal.signal_type} for {signal.symbol}: {e}")
            raise
    
    async def _update_positions(self):
        """Update current positions with latest prices"""
        if not self.positions:
            return
        
        try:
            primary_exchange = list(self.exchanges.values())[0]
            current_prices = {}
            
            # Get current prices for all position symbols
            for symbol in self.positions.keys():
                try:
                    ticker = await primary_exchange.get_ticker(symbol)
                    if ticker and 'c' in ticker:  # 'c' is close price in Kraken
                        current_prices[symbol] = Decimal(str(ticker['c'][0]))
                except Exception as e:
                    self.logger.warning(f"Error getting price for {symbol}: {e}")
            
            # Update positions with current prices
            if current_prices:
                # Update strategies
                for strategy in self.strategies.values():
                    strategy.update_positions(current_prices)
                
                # Update our position tracking
                for symbol, position in self.positions.items():
                    if symbol in current_prices:
                        position.current_price = current_prices[symbol]
                        
                        # Calculate PnL
                        if position.side == 'long':
                            position.pnl = (position.current_price - position.entry_price) * position.quantity
                        else:  # short
                            position.pnl = (position.entry_price - position.current_price) * position.quantity
                        
                        position.pnl_percentage = float(position.pnl / (position.entry_price * position.quantity)) * 100
        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _check_position_exits(self):
        """Check if any positions should be closed"""
        if not self.positions:
            return
        
        positions_to_close = []
        primary_exchange = list(self.exchanges.values())[0]
        
        for symbol, position in self.positions.items():
            try:
                # Get latest market data for exit decision
                df = self.data_collector.get_ohlcv_data(symbol, limit=10)
                if df.empty:
                    continue
                
                latest_data = df.iloc[-1]
                
                # Check each strategy's exit conditions
                should_exit = False
                exit_reason = ""
                
                for strategy in self.strategies.values():
                    if strategy.should_exit_position(position, latest_data):
                        should_exit = True
                        exit_reason = f"Strategy {strategy.name} exit signal"
                        break
                
                # Risk manager override
                if not should_exit:
                    # Check portfolio-level risk
                    metrics = self.risk_manager.update_portfolio_metrics(self.balance, self.positions)
                    if self.risk_manager.should_reduce_exposure(metrics):
                        should_exit = True
                        exit_reason = "Risk manager: reducing exposure"
                
                if should_exit:
                    positions_to_close.append((symbol, exit_reason))
            
            except Exception as e:
                self.logger.error(f"Error checking exit for {symbol}: {e}")
        
        # Close positions that meet exit criteria
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, reason)
    
    async def _close_position(self, symbol: str, reason: str = "Manual close"):
        """Close a specific position"""
        if symbol not in self.positions:
            self.logger.warning(f"Attempted to close non-existent position: {symbol}")
            return
        
        position = self.positions[symbol]
        primary_exchange = list(self.exchanges.values())[0]
        
        try:
            # Determine order side for closing
            close_side = "sell" if position.side == 'long' else "buy"
            
            # Place closing order
            result = await primary_exchange.place_order(
                symbol=symbol,
                side=close_side,
                amount=position.quantity,
                order_type="market"
            )
            
            if result and 'txid' in result:
                # Remove from positions
                closed_position = self.positions.pop(symbol)
                
                self.logger.info(f"Closed {position.side.upper()} position: {symbol} - "
                               f"PnL: {position.pnl} ({position.pnl_percentage:.2f}%) - Reason: {reason}")
                
                # Update performance tracking
                self._record_closed_position(closed_position, reason)
        
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    def _record_closed_position(self, position: Position, reason: str):
        """Record closed position for performance tracking"""
        closed_position_record = {
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': float(position.entry_price),
            'exit_price': float(position.current_price),
            'quantity': float(position.quantity),
            'pnl': float(position.pnl),
            'pnl_percentage': position.pnl_percentage,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'hold_duration': (datetime.now() - position.entry_time).total_seconds(),
            'exit_reason': reason
        }
        
        # Store in performance history
        if 'closed_positions' not in self.performance_stats:
            self.performance_stats['closed_positions'] = []
        
        self.performance_stats['closed_positions'].append(closed_position_record)
    
    async def _close_all_positions(self):
        """Close all open positions"""
        if not self.positions:
            return
        
        self.logger.info(f"Closing all {len(self.positions)} positions...")
        
        # Create a copy to avoid modification during iteration
        positions_to_close = list(self.positions.keys())
        
        for symbol in positions_to_close:
            await self._close_position(symbol, "Bot shutdown")
    
    def _should_close_positions_on_stop(self) -> bool:
        """Determine if positions should be closed when bot stops"""
        # This is a policy decision - you might want to keep positions open
        # or only close them in emergency situations
        return self.risk_manager.emergency_stop
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            # Calculate basic stats
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            open_positions = len(self.positions)
            
            # Calculate unrealized PnL percentage
            total_position_value = sum(pos.entry_price * pos.quantity for pos in self.positions.values())
            unrealized_pnl_pct = float(total_pnl / total_position_value * 100) if total_position_value > 0 else 0.0
            
            self.performance_stats.update({
                'last_update': datetime.now(),
                'balance': float(self.balance),
                'open_positions': open_positions,
                'total_unrealized_pnl': float(total_pnl),
                'unrealized_pnl_percentage': unrealized_pnl_pct,
                'risk_status': self.risk_manager.get_risk_status()
            })
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")
    
    async def _save_performance_report(self):
        """Save final performance report"""
        try:
            report_path = Path("logs") / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Generate comprehensive report
            portfolio_metrics = self.risk_manager.calculate_portfolio_metrics()
            
            report = {
                'bot_runtime': {
                    'start_time': getattr(self, 'start_time', datetime.now()),
                    'end_time': datetime.now(),
                    'total_runtime_hours': 0  # Would calculate actual runtime
                },
                'final_stats': self.performance_stats,
                'portfolio_metrics': portfolio_metrics,
                'configuration': {
                    'strategies': list(self.strategies.keys()),
                    'exchanges': list(self.exchanges.keys()),
                    'risk_limits': self.risk_manager.get_risk_status()
                }
            }
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update_time,
            'balance': float(self.balance),
            'positions': {symbol: {
                'side': pos.side,
                'entry_price': float(pos.entry_price),
                'current_price': float(pos.current_price),
                'quantity': float(pos.quantity),
                'pnl': float(pos.pnl),
                'pnl_percentage': pos.pnl_percentage,
                'entry_time': pos.entry_time,
                'hold_duration_hours': (datetime.now() - pos.entry_time).total_seconds() / 3600
            } for symbol, pos in self.positions.items()},
            'strategies': {name: strategy.get_strategy_stats() 
                          for name, strategy in self.strategies.items()},
            'risk_status': self.risk_manager.get_risk_status() if self.risk_manager else {},
            'performance': self.performance_stats
        }
    
    async def manual_close_position(self, symbol: str) -> bool:
        """Manually close a specific position"""
        if symbol in self.positions:
            await self._close_position(symbol, "Manual close")
            return True
        return False
    
    async def manual_close_all_positions(self) -> int:
        """Manually close all positions"""
        closed_count = len(self.positions)
        await self._close_all_positions()
        return closed_count

# Main entry point
async def main():
    """Main entry point for the trading bot"""
    bot = None
    try:
        # Create and start bot
        bot = TradingBot()
        
        # Store start time for reporting
        bot.start_time = datetime.now()
        
        await bot.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user...")
        if bot:
            await bot.stop()
    except Exception as e:
        print(f"Bot error: {e}")
        if bot:
            await bot.stop()

if __name__ == "__main__":
    print("Starting Trading Bot...")
    asyncio.run(main())