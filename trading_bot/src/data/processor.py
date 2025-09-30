# src/data/processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

class DataProcessor:
    """Process and prepare market data for trading strategies and ML models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}  # Store scalers for different symbols/features
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators"""
        if df.empty:
            return df
        
        data = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Convert to numpy arrays for talib
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data['volume'].values
        
        try:
            # Moving Averages
            data['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            data['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            data['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            data['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            data['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # MACD
            data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(close_prices)
            
            # RSI
            data['rsi'] = talib.RSI(close_prices, timeperiod=14)
            
            # Bollinger Bands
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(close_prices)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (close_prices - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Stochastic
            data['stoch_k'], data['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            
            # ADX (Average Directional Index)
            data['adx'] = talib.ADX(high_prices, low_prices, close_prices)
            data['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
            data['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
            
            # Volume indicators
            data['volume_sma'] = talib.SMA(volume, timeperiod=20)
            data['volume_ratio'] = volume / data['volume_sma']
            
            # Price-based features
            data['price_change'] = data['close'].pct_change()
            data['price_range'] = (data['high'] - data['low']) / data['close']
            data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            
            # Volatility
            data['volatility_10'] = data['price_change'].rolling(window=10).std()
            data['volatility_20'] = data['price_change'].rolling(window=20).std()
            
            # Support and Resistance levels (simplified)
            data['resistance'] = data['high'].rolling(window=20).max()
            data['support'] = data['low'].rolling(window=20).min()
            data['distance_to_resistance'] = (data['resistance'] - data['close']) / data['close']
            data['distance_to_support'] = (data['close'] - data['support']) / data['close']
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise
        
        return data
    
    def calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom features for ML models"""
        data = df.copy()
        
        # Momentum features
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # Mean reversion features
        for period in [10, 20, 50]:
            data[f'mean_reversion_{period}'] = (data['close'] - data['close'].rolling(period).mean()) / data['close'].rolling(period).std()
        
        # Trend strength
        data['trend_strength'] = abs(data['close'].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0]))
        
        # Price patterns (simplified)
        data['higher_high'] = ((data['high'] > data['high'].shift(1)) & 
                              (data['high'].shift(1) > data['high'].shift(2))).astype(int)
        data['lower_low'] = ((data['low'] < data['low'].shift(1)) & 
                            (data['low'].shift(1) < data['low'].shift(2))).astype(int)
        
        # Volume-price relationship
        data['vp_trend'] = np.where(data['price_change'] > 0, 
                                   data['volume'], -data['volume'])
        data['volume_price_trend'] = data['vp_trend'].rolling(10).sum()
        
        return data
    
    def prepare_ml_features(self, df: pd.DataFrame, target_column: str = 'close', 
                           lookback_periods: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for machine learning models"""
        data = df.copy()
        
        # Calculate technical indicators if not already present
        if 'sma_10' not in data.columns:
            data = self.calculate_technical_indicators(data)
            data = self.calculate_custom_features(data)
        
        # Select feature columns (exclude non-numeric and target-related columns)
        feature_columns = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume'] 
                          and data[col].dtype in ['float64', 'int64']]
        
        # Remove rows with NaN values
        data_clean = data[feature_columns + [target_column]].dropna()
        
        if len(data_clean) < lookback_periods + 1:
            raise ValueError(f"Not enough data. Need at least {lookback_periods + 1} rows")
        
        # Create sequences for LSTM/time-series models
        X, y = [], []
        
        for i in range(lookback_periods, len(data_clean)):
            # Features: use multiple time steps
            feature_sequence = data_clean[feature_columns].iloc[i-lookback_periods:i].values
            X.append(feature_sequence)
            
            # Target: predict next period's price change
            current_price = data_clean[target_column].iloc[i-1]
            next_price = data_clean[target_column].iloc[i]
            price_change = (next_price - current_price) / current_price
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, X: np.ndarray, scaler_name: str = "default", 
                          fit_scaler: bool = True) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if fit_scaler or scaler_name not in self.scalers:
            self.scalers[scaler_name] = StandardScaler()
            # Reshape for scaling if needed (for 3D arrays from LSTM)
            if X.ndim == 3:
                original_shape = X.shape
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scalers[scaler_name].fit_transform(X_reshaped)
                return X_scaled.reshape(original_shape)
            else:
                return self.scalers[scaler_name].fit_transform(X)
        else:
            if X.ndim == 3:
                original_shape = X.shape
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scalers[scaler_name].transform(X_reshaped)
                return X_scaled.reshape(original_shape)
            else:
                return self.scalers[scaler_name].transform(X)
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 5, 
                     threshold: float = 0.01) -> np.ndarray:
        """Create classification labels for price movement prediction"""
        price_changes = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Create three classes: -1 (down), 0 (sideways), 1 (up)
        labels = np.where(price_changes > threshold, 1,
                         np.where(price_changes < -threshold, -1, 0))
        
        return labels
    
    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Detect and optionally remove outliers"""
        data = df.copy()
        
        if method == "iqr":
            Q1 = data['close'].quantile(0.25)
            Q3 = data['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data['is_outlier'] = (data['close'] < lower_bound) | (data['close'] > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data['close']))
            data['is_outlier'] = z_scores > 3
        
        return data
    
    def resample_data(self, df: pd.DataFrame, freq: str = "5T") -> pd.DataFrame:
        """Resample OHLCV data to different timeframes"""
        if df.empty:
            return df
        
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced volatility metrics"""
        data = df.copy()
        
        # True Range
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Average True Range
        data['atr_14'] = data['tr'].rolling(window=14).mean()
        
        # Garman-Klass volatility estimator
        data['gk_volatility'] = np.log(data['high'] / data['low']) ** 2 / 2 - (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2
        data['gk_volatility_ma'] = data['gk_volatility'].rolling(window=20).mean()
        
        # Parkinson volatility estimator
        data['parkinson_vol'] = (np.log(data['high'] / data['low']) ** 2) / (4 * np.log(2))
        data['parkinson_vol_ma'] = data['parkinson_vol'].rolling(window=20).mean()
        
        return data
    
    def calculate_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure indicators"""
        data = df.copy()
        
        # Price impact approximation
        data['price_impact'] = abs(data['price_change']) / np.log(1 + data['volume'])
        
        # Roll measure (bid-ask spread estimator)
        price_changes = data['close'].diff()
        data['roll_measure'] = 2 * np.sqrt(-np.cov(price_changes[:-1], price_changes[1:])[0, 1])
        
        # Volume-synchronized probability of informed trading (VPIN)
        data['vpin'] = abs(data['volume'] * data['price_change']).rolling(window=50).sum() / data['volume'].rolling(window=50).sum()
        
        return data

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    high_prices = close_prices + np.abs(np.random.randn(1000) * 50)
    low_prices = close_prices - np.abs(np.random.randn(1000) * 50)
    open_prices = np.concatenate([[close_prices[0]], close_prices[:-1]]) + np.random.randn(1000) * 20
    volumes = np.abs(np.random.randn(1000) * 1000) + 1000
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test data processor
    processor = DataProcessor(config=None)
    
    try:
        # Calculate technical indicators
        df_processed = processor.calculate_technical_indicators(df)
        print(f"Added {len(df_processed.columns) - len(df.columns)} technical indicators")
        
        # Calculate custom features
        df_features = processor.calculate_custom_features(df_processed)
        print(f"Total features: {len(df_features.columns)}")
        
        # Prepare ML features
        X, y = processor.prepare_ml_features(df_features)
        print(f"ML features shape: {X.shape}, targets shape: {y.shape}")
        
        # Test normalization
        X_normalized = processor.normalize_features(X)
        print(f"Features normalized, mean: {X_normalized.mean():.4f}, std: {X_normalized.std():.4f}")
        
    except Exception as e:
        print(f"Error testing data processor: {e}")