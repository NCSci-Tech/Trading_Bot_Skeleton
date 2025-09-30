use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rayon::prelude::*;
use ndarray::{Array1, ArrayView1, s};

/// Calculate Simple Moving Average
#[pyfunction]
fn sma(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    
    if period > len || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid period"));
    }
    
    let mut result = vec![f64::NAN; len];
    
    for i in (period - 1)..len {
        let sum: f64 = prices.slice(s![i + 1 - period..=i]).sum();
        result[i] = sum / period as f64;
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate Exponential Moving Average
#[pyfunction]
fn ema(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    
    if len == 0 || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid input"));
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; len];
    
    // Initialize with first valid price
    let mut ema_value = prices[0];
    result[0] = ema_value;
    
    for i in 1..len {
        ema_value = alpha * prices[i] + (1.0 - alpha) * ema_value;
        result[i] = ema_value;
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate RSI (Relative Strength Index)
#[pyfunction]
fn rsi(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    
    if period >= len || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid period"));
    }
    
    let mut result = vec![f64::NAN; len];
    
    // Calculate price changes
    let mut gains = Vec::with_capacity(len);
    let mut losses = Vec::with_capacity(len);
    
    gains.push(0.0);
    losses.push(0.0);
    
    for i in 1..len {
        let change = prices[i] - prices[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });
    }
    
    // Calculate initial averages using SMA
    if len > period {
        let initial_avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let initial_avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;
        
        let mut avg_gain = initial_avg_gain;
        let mut avg_loss = initial_avg_loss;
        
        // Calculate RSI for initial period
        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 0.0 };
        result[period] = 100.0 - (100.0 / (1.0 + rs));
        
        // Use Wilder's smoothing for subsequent values
        let alpha = 1.0 / period as f64;
        
        for i in (period + 1)..len {
            avg_gain = (1.0 - alpha) * avg_gain + alpha * gains[i];
            avg_loss = (1.0 - alpha) * avg_loss + alpha * losses[i];
            
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 0.0 };
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate MACD (Moving Average Convergence Divergence)
#[pyfunction]
fn macd(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    fast: usize,
    slow: usize,
    signal: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    
    // Calculate fast and slow EMAs
    let ema_fast = ema(py, prices, fast)?;
    let ema_slow = ema(py, prices, slow)?;
    
    let fast_array = ema_fast.as_ref(py).readonly().as_array();
    let slow_array = ema_slow.as_ref(py).readonly().as_array();
    
    // Calculate MACD line
    let macd_line: Vec<f64> = fast_array.iter().zip(slow_array.iter())
        .map(|(fast_val, slow_val)| {
            if fast_val.is_nan() || slow_val.is_nan() {
                f64::NAN
            } else {
                fast_val - slow_val
            }
        })
        .collect();
    
    // Calculate signal line (EMA of MACD line)
    let macd_array = macd_line.clone().into_pyarray(py);
    let signal_line = ema(py, macd_array.readonly(), signal)?;
    
    // Calculate histogram
    let signal_readonly = signal_line.as_ref(py).readonly().as_array();
    let histogram: Vec<f64> = macd_line.iter().zip(signal_readonly.iter())
        .map(|(macd_val, signal_val)| {
            if macd_val.is_nan() || signal_val.is_nan() {
                f64::NAN
            } else {
                macd_val - signal_val
            }
        })
        .collect();
    
    Ok((
        macd_line.into_pyarray(py).to_owned(),
        signal_line,
        histogram.into_pyarray(py).to_owned(),
    ))
}

/// Calculate Bollinger Bands
#[pyfunction]
fn bollinger_bands(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    
    let prices = prices.as_array();
    let len = prices.len();
    
    if period > len || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid period"));
    }
    
    let mut upper = vec![f64::NAN; len];
    let mut middle = vec![f64::NAN; len];
    let mut lower = vec![f64::NAN; len];
    
    for i in (period - 1)..len {
        let window = prices.slice(s![i + 1 - period..=i]);
        let mean = window.sum() / period as f64;
        
        // Calculate standard deviation
        let variance: f64 = window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        
        middle[i] = mean;
        upper[i] = mean + (std_dev * std);
        lower[i] = mean - (std_dev * std);
    }
    
    Ok((
        upper.into_pyarray(py).to_owned(),
        middle.into_pyarray(py).to_owned(),
        lower.into_pyarray(py).to_owned(),
    ))
}

/// Calculate Stochastic Oscillator
#[pyfunction]
fn stochastic(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    k_period: usize,
    d_period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    
    if len != low.len() || len != close.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Arrays must have same length"));
    }
    
    if k_period > len || k_period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid k_period"));
    }
    
    let mut k_values = vec![f64::NAN; len];
    
    // Calculate %K
    for i in (k_period - 1)..len {
        let window_high = high.slice(s![i + 1 - k_period..=i]);
        let window_low = low.slice(s![i + 1 - k_period..=i]);
        
        let highest = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest != lowest {
            k_values[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0;
        } else {
            k_values[i] = 50.0; // Neutral value when no range
        }
    }
    
    // Calculate %D (SMA of %K)
    let k_array = k_values.clone().into_pyarray(py);
    let d_values = sma(py, k_array.readonly(), d_period)?;
    
    Ok((
        k_values.into_pyarray(py).to_owned(),
        d_values,
    ))
}

/// Calculate Average True Range (ATR)
#[pyfunction]
fn atr(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    
    if len != low.len() || len != close.len() || len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid input arrays"));
    }
    
    // Calculate True Range
    let mut tr_values = vec![f64::NAN; len];
    tr_values[0] = high[0] - low[0]; // First value
    
    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        
        tr_values[i] = hl.max(hc).max(lc);
    }
    
    // Calculate ATR using Wilder's smoothing
    let tr_array = tr_values.into_pyarray(py);
    let atr_values = ema(py, tr_array.readonly(), period * 2 - 1)?; // Wilder's smoothing approximation
    
    Ok(atr_values)
}

/// Parallel batch processing of multiple indicators
#[pyfunction]
fn batch_indicators(py: Python, prices: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    let prices_vec: Vec<f64> = prices.as_array().to_vec();
    let len = prices_vec.len();
    
    if len < 50 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need at least 50 data points"));
    }
    
    // Process indicators in parallel
    let results: Vec<_> = [5, 10, 20, 50].par_iter().filter_map(|&period| {
        if period < len {
            let mut sma_result = vec![f64::NAN; len];
            
            for i in (period - 1)..len {
                let sum: f64 = prices_vec[i + 1 - period..=i].iter().sum();
                sma_result[i] = sum / period as f64;
            }
            
            Some((format!("sma_{}", period), sma_result))
        } else {
            None
        }
    }).collect();
    
    // Convert to Python dictionary
    let dict = pyo3::types::PyDict::new(py);
    
    for (name, values) in results {
        dict.set_item(name, values.into_pyarray(py))?;
    }
    
    // Add RSI
    if len >= 15 {
        let rsi_result = rsi(py, prices, 14)?;
        dict.set_item("rsi_14", rsi_result)?;
    }
    
    // Add EMAs
    for &period in &[12, 26] {
        if period < len {
            let ema_result = ema(py, prices, period)?;
            dict.set_item(format!("ema_{}", period), ema_result)?;
        }
    }
    
    Ok(dict.into())
}

/// Calculate multiple timeframe analysis
#[pyfunction]
fn multi_timeframe_sma(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    periods: Vec<usize>,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    
    for period in periods {
        let sma_result = sma(py, prices, period)?;
        dict.set_item(format!("sma_{}", period), sma_result)?;
    }
    
    Ok(dict.into())
}

/// Calculate price momentum
#[pyfunction]
fn momentum(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    
    if period >= len || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid period"));
    }
    
    let mut result = vec![f64::NAN; len];
    
    for i in period..len {
        if prices[i - period] != 0.0 {
            result[i] = prices[i] / prices[i - period];
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Calculate Rate of Change (ROC)
#[pyfunction]
fn roc(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    
    if period >= len || period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid period"));
    }
    
    let mut result = vec![f64::NAN; len];
    
    for i in period..len {
        if prices[i - period] != 0.0 {
            result[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100.0;
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

#[pymodule]
fn indicators(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(atr, m)?)?;
    m.add_function(wrap_pyfunction!(batch_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(multi_timeframe_sma, m)?)?;
    m.add_function(wrap_pyfunction!(momentum, m)?)?;
    m.add_function(wrap_pyfunction!(roc, m)?)?;
    Ok(())
}