use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use chrono::{DateTime, Utc};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: BTreeMap<OrderedFloat, OrderBookLevel>, // Price as key (inverted for bids)
    pub asks: BTreeMap<OrderedFloat, OrderBookLevel>, // Price as key
    pub last_update: DateTime<Utc>,
    pub symbol: String,
}

// Wrapper for f64 to enable ordering in BTreeMap
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update: Utc::now(),
            symbol,
        }
    }
    
    pub fn update_bid(&mut self, price: f64, size: f64) {
        let price_key = OrderedFloat(-price); // Negative for reverse ordering (highest first)
        
        if size == 0.0 {
            self.bids.remove(&price_key);
        } else {
            self.bids.insert(
                price_key,
                OrderBookLevel {
                    price,
                    size,
                    timestamp: Utc::now(),
                }
            );
        }
        self.last_update = Utc::now();
    }
    
    pub fn update_ask(&mut self, price: f64, size: f64) {
        let price_key = OrderedFloat(price);
        
        if size == 0.0 {
            self.asks.remove(&price_key);
        } else {
            self.asks.insert(
                price_key,
                OrderBookLevel {
                    price,
                    size,
                    timestamp: Utc::now(),
                }
            );
        }
        self.last_update = Utc::now();
    }
    
    pub fn get_best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.values().next()
    }
    
    pub fn get_best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.values().next()
    }
    
    pub fn get_spread(&self) -> Option<f64> {
        match (self.get_best_bid(), self.get_best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }
    
    pub fn get_mid_price(&self) -> Option<f64> {
        match (self.get_best_bid(), self.get_best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }
    
    pub fn get_depth(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids: Vec<(f64, f64)> = self.bids.values()
            .take(levels)
            .map(|level| (level.price, level.size))
            .collect();
        
        let asks: Vec<(f64, f64)> = self.asks.values()
            .take(levels)
            .map(|level| (level.price, level.size))
            .collect();
        
        (bids, asks)
    }
    
    pub fn get_volume_at_price(&self, price: f64, side: &str) -> Option<f64> {
        match side {
            "bid" => {
                let key = OrderedFloat(-price);
                self.bids.get(&key).map(|level| level.size)
            },
            "ask" => {
                let key = OrderedFloat(price);
                self.asks.get(&key).map(|level| level.size)
            },
            _ => None,
        }
    }
    
    pub fn calculate_imbalance(&self, levels: usize) -> f64 {
        let (bids, asks) = self.get_depth(levels);
        
        let bid_volume: f64 = bids.iter().map(|(_, vol)| vol).sum();
        let ask_volume: f64 = asks.iter().map(|(_, vol)| vol).sum();
        
        if bid_volume + ask_volume == 0.0 {
            0.0
        } else {
            (bid_volume - ask_volume) / (bid_volume + ask_volume)
        }
    }
    
    pub fn estimate_market_impact(&self, quantity: f64, side: &str) -> Option<f64> {
        let levels = match side {
            "buy" => &self.asks,
            "sell" => &self.bids,
            _ => return None,
        };
        
        let mut remaining_quantity = quantity;
        let mut total_cost = 0.0;
        let mut shares_filled = 0.0;
        
        for level in levels.values() {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            let fill_quantity = remaining_quantity.min(level.size);
            total_cost += fill_quantity * level.price;
            shares_filled += fill_quantity;
            remaining_quantity -= fill_quantity;
        }
        
        if shares_filled > 0.0 {
            Some(total_cost / shares_filled)
        } else {
            None
        }
    }
}

#[pyclass]
pub struct PyOrderBook {
    inner: OrderBook,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    pub fn new(symbol: String) -> Self {
        Self {
            inner: OrderBook::new(symbol),
        }
    }
    
    pub fn update_bid(&mut self, price: f64, size: f64) {
        self.inner.update_bid(price, size);
    }
    
    pub fn update_ask(&mut self, price: f64, size: f64) {
        self.inner.update_ask(price, size);
    }
    
    pub fn get_best_bid(&self) -> Option<(f64, f64)> {
        self.inner.get_best_bid().map(|level| (level.price, level.size))
    }
    
    pub fn get_best_ask(&self) -> Option<(f64, f64)> {
        self.inner.get_best_ask().map(|level| (level.price, level.size))
    }
    
    pub fn get_spread(&self) -> Option<f64> {
        self.inner.get_spread()
    }
    
    pub fn get_mid_price(&self) -> Option<f64> {
        self.inner.get_mid_price()
    }
    
    pub fn get_depth(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        self.inner.get_depth(levels)
    }
    
    pub fn get_volume_at_price(&self, price: f64, side: &str) -> Option<f64> {
        self.inner.get_volume_at_price(price, side)
    }
    
    pub fn calculate_imbalance(&self, levels: usize) -> f64 {
        self.inner.calculate_imbalance(levels)
    }
    
    pub fn estimate_market_impact(&self, quantity: f64, side: &str) -> Option<f64> {
        self.inner.estimate_market_impact(quantity, side)
    }
    
    pub fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        
        let (bids, asks) = self.inner.get_depth(10);
        dict.set_item("bids", bids)?;
        dict.set_item("asks", asks)?;
        dict.set_item("spread", self.inner.get_spread())?;
        dict.set_item("mid_price", self.inner.get_mid_price())?;
        dict.set_item("last_update", self.inner.last_update.timestamp())?;
        dict.set_item("symbol", &self.inner.symbol)?;
        dict.set_item("bid_levels", self.inner.bids.len())?;
        dict.set_item("ask_levels", self.inner.asks.len())?;
        
        Ok(dict.into())
    }
    
    pub fn get_levels_info(&self) -> (usize, usize) {
        (self.inner.bids.len(), self.inner.asks.len())
    }
    
    pub fn clear(&mut self) {
        self.inner.bids.clear();
        self.inner.asks.clear();
        self.inner.last_update = Utc::now();
    }
}

/// Calculate volume-weighted average price (VWAP)
#[pyfunction]
pub fn calculate_vwap(prices: Vec<f64>, volumes: Vec<f64>) -> PyResult<f64> {
    if prices.len() != volumes.len() || prices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid input lengths"));
    }
    
    let total_volume: f64 = volumes.iter().sum();
    if total_volume == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Total volume is zero"));
    }
    
    let weighted_sum: f64 = prices.iter().zip(volumes.iter())
        .map(|(price, volume)| price * volume)
        .sum();
    
    Ok(weighted_sum / total_volume)
}

/// Calculate time-weighted average price (TWAP)
#[pyfunction] 
pub fn calculate_twap(prices: Vec<f64>, timestamps: Vec<i64>) -> PyResult<f64> {
    if prices.len() != timestamps.len() || prices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid input"));
    }
    
    let mut weighted_sum = 0.0;
    let mut total_time = 0_i64;
    
    for i in 1..prices.len() {
        let time_diff = timestamps[i] - timestamps[i-1];
        weighted_sum += prices[i-1] * time_diff as f64;
        total_time += time_diff;
    }
    
    if total_time == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("No time difference"));
    }
    
    Ok(weighted_sum / total_time as f64)
}

/// Calculate order book pressure
#[pyfunction]
pub fn calculate_book_pressure(
    bid_prices: Vec<f64>, 
    bid_volumes: Vec<f64>,
    ask_prices: Vec<f64>, 
    ask_volumes: Vec<f64>,
    depth_levels: usize
) -> PyResult<f64> {
    
    if bid_prices.len() != bid_volumes.len() || ask_prices.len() != ask_volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Price and volume arrays must match"));
    }
    
    let levels = depth_levels.min(bid_prices.len()).min(ask_prices.len());
    
    if levels == 0 {
        return Ok(0.0);
    }
    
    let bid_pressure: f64 = bid_volumes.iter().take(levels).sum();
    let ask_pressure: f64 = ask_volumes.iter().take(levels).sum();
    
    if bid_pressure + ask_pressure == 0.0 {
        Ok(0.0)
    } else {
        Ok((bid_pressure - ask_pressure) / (bid_pressure + ask_pressure))
    }
}

/// Calculate liquidity score
#[pyfunction]
pub fn calculate_liquidity_score(
    bid_volumes: Vec<f64>,
    ask_volumes: Vec<f64>,
    spread: f64,
    mid_price: f64
) -> PyResult<f64> {
    
    if spread <= 0.0 || mid_price <= 0.0 {
        return Ok(0.0);
    }
    
    let total_bid_volume: f64 = bid_volumes.iter().sum();
    let total_ask_volume: f64 = ask_volumes.iter().sum();
    let total_volume = total_bid_volume + total_ask_volume;
    
    if total_volume == 0.0 {
        return Ok(0.0);
    }
    
    // Normalized spread (lower is better)
    let spread_score = 1.0 / (1.0 + (spread / mid_price) * 1000.0);
    
    // Volume score (higher is better)
    let volume_score = (total_volume / 1000.0).min(1.0);
    
    // Combined liquidity score
    Ok((spread_score + volume_score) / 2.0)
}

/// Detect price levels with significant volume
#[pyfunction]
pub fn detect_support_resistance(
    prices: Vec<f64>,
    volumes: Vec<f64>,
    min_volume_threshold: f64
) -> PyResult<Vec<f64>> {
    
    if prices.len() != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have same length"));
    }
    
    let mut significant_levels = Vec::new();
    
    for (price, volume) in prices.iter().zip(volumes.iter()) {
        if *volume >= min_volume_threshold {
            significant_levels.push(*price);
        }
    }
    
    // Sort and deduplicate similar price levels
    significant_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Group similar prices (within 0.1% of each other)
    let mut filtered_levels = Vec::new();
    let mut last_price = -1.0;
    
    for price in significant_levels {
        if last_price < 0.0 || (price - last_price).abs() / last_price > 0.001 {
            filtered_levels.push(price);
            last_price = price;
        }
    }
    
    Ok(filtered_levels)
}

/// Calculate order flow imbalance
#[pyfunction]
pub fn calculate_order_flow_imbalance(
    buy_volumes: Vec<f64>,
    sell_volumes: Vec<f64>,
    window_size: usize
) -> PyResult<Vec<f64>> {
    
    if buy_volumes.len() != sell_volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have same length"));
    }
    
    let len = buy_volumes.len();
    let mut imbalance = vec![0.0; len];
    
    for i in 0..len {
        let start = if i >= window_size { i - window_size + 1 } else { 0 };
        let end = i + 1;
        
        let buy_sum: f64 = buy_volumes[start..end].iter().sum();
        let sell_sum: f64 = sell_volumes[start..end].iter().sum();
        
        if buy_sum + sell_sum > 0.0 {
            imbalance[i] = (buy_sum - sell_sum) / (buy_sum + sell_sum);
        }
    }
    
    Ok(imbalance)
}

#[pymodule]
fn orderbook(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOrderBook>()?;
    m.add_function(wrap_pyfunction!(calculate_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_twap, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_book_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_liquidity_score, m)?)?;
    m.add_function(wrap_pyfunction!(detect_support_resistance, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_order_flow_imbalance, m)?)?;
    Ok(())
}