use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, NaiveDateTime};
use std::collections::HashMap;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: String, // "buy" or "sell"
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub trade_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub entry_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_trade_duration: f64, // in hours
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    pub calmar_ratio: f64,
    pub sortino_ratio: f64,
    pub equity_curve: Vec<f64>,
    pub drawdown_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub trade_pnls: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: f64, // in days
    pub win_rate: f64,
    pub profit_factor: f64,
    pub recovery_factor: f64,
    pub expectancy: f64,
}

#[pyclass]
pub struct BacktestEngine {
    initial_capital: f64,
    current_capital: f64,
    positions: HashMap<String, Position>,
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
    timestamps: Vec<DateTime<Utc>>,
    commission_rate: f64,
    slippage: f64,
    trade_counter: usize,
}

#[pymethods]
impl BacktestEngine {
    #[new]
    pub fn new(initial_capital: f64, commission_rate: f64, slippage: f64) -> Self {
        Self {
            initial_capital,
            current_capital: initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            timestamps: vec![Utc::now()],
            commission_rate,
            slippage,
            trade_counter: 0,
        }
    }
    
    pub fn execute_trade(
        &mut self, 
        timestamp: &str, 
        symbol: &str, 
        side: &str, 
        price: f64, 
        quantity: f64
    ) -> PyResult<()> {
        let timestamp = DateTime::parse_from_rfc3339(timestamp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        if quantity <= 0.0 || price <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid price or quantity"));
        }
        
        // Apply slippage
        let adjusted_price = match side {
            "buy" => price * (1.0 + self.slippage),
            "sell" => price * (1.0 - self.slippage),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid side")),
        };
        
        // Calculate commission
        let commission = adjusted_price * quantity * self.commission_rate;
        let trade_value = adjusted_price * quantity;
        
        // Update or create position
        let position = self.positions.entry(symbol.to_string()).or_insert(Position {
            symbol: symbol.to_string(),
            quantity: 0.0,
            avg_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            entry_time: timestamp,
            last_update: timestamp,
        });
        
        let mut realized_pnl = 0.0;
        
        match side {
            "buy" => {
                if position.quantity < 0.0 {
                    // Closing short position
                    let close_quantity = quantity.min(-position.quantity);
                    realized_pnl = (position.avg_price - adjusted_price) * close_quantity;
                    position.realized_pnl += realized_pnl;
                    position.quantity += close_quantity;
                    
                    if quantity > close_quantity {
                        // Opening new long position
                        let new_quantity = quantity - close_quantity;
                        position.avg_price = adjusted_price;
                        position.quantity = new_quantity;
                        position.entry_time = timestamp;
                    }
                } else {
                    // Adding to long position or opening new long
                    if position.quantity == 0.0 {
                        position.entry_time = timestamp;
                        position.avg_price = adjusted_price;
                        position.quantity = quantity;
                    } else {
                        let total_value = position.avg_price * position.quantity + trade_value;
                        let total_quantity = position.quantity + quantity;
                        position.avg_price = total_value / total_quantity;
                        position.quantity = total_quantity;
                    }
                }
                self.current_capital -= trade_value + commission;
            }
            "sell" => {
                if position.quantity > 0.0 {
                    // Closing long position
                    let close_quantity = quantity.min(position.quantity);
                    realized_pnl = (adjusted_price - position.avg_price) * close_quantity;
                    position.realized_pnl += realized_pnl;
                    position.quantity -= close_quantity;
                    
                    if quantity > close_quantity {
                        // Opening new short position
                        let new_quantity = quantity - close_quantity;
                        position.avg_price = adjusted_price;
                        position.quantity = -new_quantity;
                        position.entry_time = timestamp;
                    }
                } else {
                    // Adding to short position or opening new short
                    if position.quantity == 0.0 {
                        position.entry_time = timestamp;
                        position.avg_price = adjusted_price;
                        position.quantity = -quantity;
                    } else {
                        let total_value = position.avg_price * (-position.quantity) + trade_value;
                        let total_quantity = -position.quantity + quantity;
                        position.avg_price = total_value / total_quantity;
                        position.quantity = -total_quantity;
                    }
                }
                self.current_capital += trade_value - commission;
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid side")),
        }
        
        position.last_update = timestamp;
        
        // Record trade
        self.trade_counter += 1;
        self.trades.push(Trade {
            timestamp,
            symbol: symbol.to_string(),
            side: side.to_string(),
            price: adjusted_price,
            quantity,
            commission,
            trade_id: format!("trade_{}", self.trade_counter),
        });
        
        Ok(())
    }
    
    pub fn update_prices(&mut self, prices: HashMap<String, f64>, timestamp: &str) -> PyResult<()> {
        let timestamp = DateTime::parse_from_rfc3339(timestamp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        // Update unrealized PnL for all positions
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(&current_price) = prices.get(symbol) {
                if position.quantity > 0.0 {
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity;
                } else if position.quantity < 0.0 {
                    position.unrealized_pnl = (position.avg_price - current_price) * (-position.quantity);
                } else {
                    position.unrealized_pnl = 0.0;
                }
                position.last_update = timestamp;
            }
        }
        
        // Update equity curve
        let total_unrealized = self.positions.values().map(|p| p.unrealized_pnl).sum::<f64>();
        let total_realized = self.positions.values().map(|p| p.realized_pnl).sum::<f64>();
        let current_equity = self.current_capital + total_unrealized + total_realized;
        
        self.equity_curve.push(current_equity);
        self.timestamps.push(timestamp);
        
        Ok(())
    }
    
    pub fn get_results(&self) -> PyResult<BacktestResults> {
        if self.equity_curve.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Insufficient data for analysis"));
        }
        
        let final_equity = *self.equity_curve.last().unwrap();
        let total_return = (final_equity - self.initial_capital) / self.initial_capital;
        
        // Calculate returns for risk metrics
        let returns: Vec<f64> = self.equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let metrics = self.calculate_performance_metrics(&returns)?;
        
        // Calculate trade-level statistics
        let trade_pnls = self.calculate_trade_pnls();
        let winning_trades = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).count();
        let losing_trades = trade_pnls.iter().filter(|&&pnl| pnl < 0.0).count();
        let total_trades = trade_pnls.len();
        
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        
        let avg_win = if winning_trades > 0 {
            trade_pnls.iter().filter(|&&pnl| pnl > 0.0).sum::<f64>() / winning_trades as f64
        } else {
            0.0
        };
        
        let avg_loss = if losing_trades > 0 {
            trade_pnls.iter().filter(|&&pnl| pnl < 0.0).sum::<f64>() / losing_trades as f64
        } else {
            0.0
        };
        
        let gross_profit: f64 = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).sum();
        let gross_loss: f64 = trade_pnls.iter().filter(|&&pnl| pnl < 0.0).sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        
        // Calculate consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_consecutive_trades(&trade_pnls);
        
        // Calculate average trade duration
        let avg_trade_duration = self.calculate_avg_trade_duration();
        
        // Calculate drawdown curve
        let drawdown_curve = self.calculate_drawdown_curve();
        
        Ok(BacktestResults {
            total_return,
            sharpe_ratio: metrics.sharpe_ratio,
            max_drawdown: metrics.max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            avg_win,
            avg_loss,
            avg_trade_duration,
            max_consecutive_wins,
            max_consecutive_losses,
            calmar_ratio: if metrics.max_drawdown > 0.0 {
                metrics.annualized_return / metrics.max_drawdown
            } else {
                0.0
            },
            sortino_ratio: self.calculate_sortino_ratio(&returns),
            equity_curve: self.equity_curve.clone(),
            drawdown_curve,
            returns,
            trade_pnls,
        })
    }
    
    pub fn reset(&mut self) {
        self.current_capital = self.initial_capital;
        self.positions.clear();
        self.trades.clear();
        self.equity_curve = vec![self.initial_capital];
        self.timestamps = vec![Utc::now()];
        self.trade_counter = 0;
    }
    
    pub fn get_current_positions(&self) -> HashMap<String, (f64, f64, f64)> {
        self.positions.iter()
            .filter(|(_, pos)| pos.quantity != 0.0)
            .map(|(symbol, pos)| {
                (symbol.clone(), (pos.quantity, pos.avg_price, pos.unrealized_pnl))
            })
            .collect()
    }
    
    pub fn get_trade_history(&self) -> Vec<(String, String, String, f64, f64, f64, String)> {
        self.trades.iter()
            .map(|trade| (
                trade.timestamp.to_rfc3339(),
                trade.symbol.clone(),
                trade.side.clone(),
                trade.price,
                trade.quantity,
                trade.commission,
                trade.trade_id.clone(),
            ))
            .collect()
    }
}

impl BacktestEngine {
    fn calculate_performance_metrics(&self, returns: &[f64]) -> PyResult<PerformanceMetrics> {
        if returns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No returns data"));
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();
        
        // Annualize metrics (assuming daily data)
        let periods_per_year = 252.0;
        let annualized_return = mean_return * periods_per_year;
        let annualized_volatility = volatility * periods_per_year.sqrt();
        
        let sharpe_ratio = if annualized_volatility > 0.0 {
            annualized_return / annualized_volatility
        } else {
            0.0
        };
        
        // Calculate maximum drawdown
        let mut peak = self.initial_capital;
        let mut max_drawdown = 0.0;
        let mut max_dd_duration = 0.0;
        let mut current_dd_duration = 0.0;
        
        for &equity in &self.equity_curve {
            if equity > peak {
                peak = equity;
                current_dd_duration = 0.0;
            } else {
                current_dd_duration += 1.0;
                max_dd_duration = max_dd_duration.max(current_dd_duration);
            }
            
            let drawdown = (peak - equity) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        let final_equity = *self.equity_curve.last().unwrap();
        let total_return_pct = (final_equity - self.initial_capital) / self.initial_capital;
        
        // Calculate other metrics
        let trade_pnls = self.calculate_trade_pnls();
        let winning_trades: Vec<f64> = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).cloned().collect();
        let losing_trades: Vec<f64> = trade_pnls.iter().filter(|&&pnl| pnl < 0.0).cloned().collect();
        
        let win_rate = if !trade_pnls.is_empty() {
            winning_trades.len() as f64 / trade_pnls.len() as f64
        } else {
            0.0
        };
        
        let gross_profit: f64 = winning_trades.iter().sum();
        let gross_loss: f64 = losing_trades.iter().sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            0.0
        };
        
        let recovery_factor = if max_drawdown > 0.0 {
            total_return_pct / max_drawdown
        } else {
            0.0
        };
        
        let expectancy = if !trade_pnls.is_empty() {
            trade_pnls.iter().sum::<f64>() / trade_pnls.len() as f64
        } else {
            0.0
        };
        
        Ok(PerformanceMetrics {
            total_pnl: final_equity - self.initial_capital,
            total_return_pct,
            annualized_return,
            volatility: annualized_volatility,
            sharpe_ratio,
            max_drawdown,
            max_drawdown_duration: max_dd_duration,
            win_rate,
            profit_factor,
            recovery_factor,
            expectancy,
        })
    }
    
    fn calculate_trade_pnls(&self) -> Vec<f64> {
        // Group trades by symbol and calculate P&L for each complete round trip
        let mut trade_pnls = Vec::new();
        let mut position_tracker: HashMap<String, (f64, f64, f64)> = HashMap::new(); // (quantity, avg_price, realized_pnl)
        
        for trade in &self.trades {
            let (mut quantity, mut avg_price, mut realized_pnl) = 
                position_tracker.get(&trade.symbol).unwrap_or(&(0.0, 0.0, 0.0)).clone();
            
            match trade.side.as_str() {
                "buy" => {
                    if quantity < 0.0 {
                        // Closing short
                        let close_qty = trade.quantity.min(-quantity);
                        let pnl = (avg_price - trade.price) * close_qty - trade.commission;
                        trade_pnls.push(pnl);
                        quantity += close_qty;
                        
                        if trade.quantity > close_qty {
                            // Opening new long
                            quantity = trade.quantity - close_qty;
                            avg_price = trade.price;
                        }
                    } else {
                        // Opening/adding to long
                        let total_value = avg_price * quantity + trade.price * trade.quantity;
                        quantity += trade.quantity;
                        avg_price = if quantity > 0.0 { total_value / quantity } else { 0.0 };
                    }
                }
                "sell" => {
                    if quantity > 0.0 {
                        // Closing long
                        let close_qty = trade.quantity.min(quantity);
                        let pnl = (trade.price - avg_price) * close_qty - trade.commission;
                        trade_pnls.push(pnl);
                        quantity -= close_qty;
                        
                        if trade.quantity > close_qty {
                            // Opening new short
                            quantity = -(trade.quantity - close_qty);
                            avg_price = trade.price;
                        }
                    } else {
                        // Opening/adding to short
                        let total_value = avg_price * (-quantity) + trade.price * trade.quantity;
                        quantity -= trade.quantity;
                        avg_price = if quantity < 0.0 { total_value / (-quantity) } else { 0.0 };
                    }
                }
                _ => {}
            }
            
            position_tracker.insert(trade.symbol.clone(), (quantity, avg_price, realized_pnl));
        }
        
        trade_pnls
    }
    
    fn calculate_consecutive_trades(&self, trade_pnls: &[f64]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;
        
        for &pnl in trade_pnls {
            if pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }
        
        (max_wins, max_losses)
    }
    
    fn calculate_avg_trade_duration(&self) -> f64 {
        if self.trades.len() < 2 {
            return 0.0;
        }
        
        let total_duration: i64 = self.trades.windows(2)
            .map(|window| {
                (window[1].timestamp - window[0].timestamp).num_seconds()
            })
            .sum();
        
        total_duration as f64 / (self.trades.len() - 1) as f64 / 3600.0 // Convert to hours
    }
    
    fn calculate_drawdown_curve(&self) -> Vec<f64> {
        let mut peak = self.initial_capital;
        let mut drawdowns = Vec::with_capacity(self.equity_curve.len());
        
        for &equity in &self.equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
            drawdowns.push(drawdown);
        }
        
        drawdowns
    }
    
    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        if downside_returns.is_empty() {
            return f64::INFINITY;
        }
        
        let downside_variance = downside_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        let downside_deviation = downside_variance.sqrt();
        
        if downside_deviation > 0.0 {
            (mean_return * 252.0) / (downside_deviation * (252.0_f64).sqrt())
        } else {
            0.0
        }
    }
}

/// Run parallel backtests with different parameters
#[pyfunction]
pub fn parallel_backtest(
    py: Python,
    initial_capital: f64,
    parameter_sets: Vec<HashMap<String, f64>>,
    trades_data: Vec<HashMap<String, String>>,
) -> PyResult<Vec<BacktestResults>> {
    
    let results: Result<Vec<BacktestResults>, _> = parameter_sets.par_iter()
        .map(|params| -> Result<BacktestResults, Box<dyn std::error::Error + Send + Sync>> {
            let mut engine = BacktestEngine::new(
                initial_capital,
                params.get("commission").unwrap_or(&0.001),
                params.get("slippage").unwrap_or(&0.0001),
            );
            
            // Execute trades
            for trade_data in &trades_data {
                if let (Some(timestamp), Some(symbol), Some(side), Some(price_str), Some(quantity_str)) = (
                    trade_data.get("timestamp"),
                    trade_data.get("symbol"),
                    trade_data.get("side"),
                    trade_data.get("price"),
                    trade_data.get("quantity"),
                ) {
                    if let (Ok(price), Ok(quantity)) = (price_str.parse::<f64>(), quantity_str.parse::<f64>()) {
                        engine.execute_trade(timestamp, symbol, side, price, quantity)?;
                    }
                }
            }
            
            engine.get_results().map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
        .collect();
    
    match results {
        Ok(results) => Ok(results),
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err("Parallel backtest failed")),
    }
}

#[pymodule]
fn backtesting(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BacktestEngine>()?;
    m.add_function(wrap_pyfunction!(parallel_backtest, m)?)?;
    Ok(())
}