# src/core/config.py
import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from decimal import Decimal

@dataclass
class ExchangeConfig:
    """Configuration for exchange connections"""
    name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # For some exchanges like Coinbase
    sandbox: bool = True
    rate_limit: int = 10  # requests per second
    timeout: int = 30

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    symbols: list[str]
    max_position_size: Decimal
    stop_loss_percentage: float
    take_profit_percentage: float

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.02  # 2% of portfolio per trade
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_drawdown: float = 0.10        # 10% max drawdown
    position_sizing_method: str = "fixed_percentage"
    emergency_stop_loss: float = 0.15  # Emergency stop at 15% loss

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "postgresql"  # postgresql, sqlite, mongodb
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = "trader"
    password: str = ""
    pool_size: int = 10

@dataclass
class AIConfig:
    """AI/ML configuration"""
    model_type: str = "lstm"  # lstm, transformer, xgboost
    feature_window: int = 50  # lookback period
    prediction_horizon: int = 5  # predict 5 periods ahead
    retrain_frequency: int = 24  # hours between retraining
    min_training_data: int = 1000  # minimum data points for training
    
@dataclass
class NotificationConfig:
    """Notification settings"""
    enabled: bool = True
    channels: list[str] = None  # email, slack, telegram, discord
    email_smtp_server: str = ""
    email_username: str = ""
    email_password: str = ""
    slack_webhook: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/config.yaml"
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.strategies: Dict[str, StrategyConfig] = {}
        self.risk: RiskConfig = RiskConfig()
        self.database: DatabaseConfig = DatabaseConfig()
        self.ai: AIConfig = AIConfig()
        self.notifications: NotificationConfig = NotificationConfig()
        
        # General settings
        self.log_level: str = "INFO"
        self.data_directory: str = "data"
        self.backup_enabled: bool = True
        self.backup_frequency: int = 24  # hours
        
        # Load configuration
        self.load_config()
        self.load_secrets()
    
    def load_config(self):
        """Load configuration from file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            self.create_default_config()
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Load exchange configurations
            for exchange_name, exchange_data in config_data.get("exchanges", {}).items():
                self.exchanges[exchange_name] = ExchangeConfig(
                    name=exchange_name,
                    **exchange_data
                )
            
            # Load strategy configurations
            for strategy_name, strategy_data in config_data.get("strategies", {}).items():
                self.strategies[strategy_name] = StrategyConfig(
                    name=strategy_name,
                    **strategy_data
                )
            
            # Load other configurations
            if "risk" in config_data:
                self.risk = RiskConfig(**config_data["risk"])
            
            if "database" in config_data:
                self.database = DatabaseConfig(**config_data["database"])
            
            if "ai" in config_data:
                self.ai = AIConfig(**config_data["ai"])
            
            if "notifications" in config_data:
                self.notifications = NotificationConfig(**config_data["notifications"])
            
            # General settings
            self.log_level = config_data.get("log_level", "INFO")
            self.data_directory = config_data.get("data_directory", "data")
            self.backup_enabled = config_data.get("backup_enabled", True)
            self.backup_frequency = config_data.get("backup_frequency", 24)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            self.create_default_config()
    
    def load_secrets(self):
        """Load sensitive data from environment variables or separate file"""
        # Override with environment variables if they exist
        for exchange_name, exchange_config in self.exchanges.items():
            env_key = f"{exchange_name.upper()}_API_KEY"
            env_secret = f"{exchange_name.upper()}_API_SECRET"
            env_passphrase = f"{exchange_name.upper()}_PASSPHRASE"
            
            if os.getenv(env_key):
                exchange_config.api_key = os.getenv(env_key)
            if os.getenv(env_secret):
                exchange_config.api_secret = os.getenv(env_secret)
            if os.getenv(env_passphrase):
                exchange_config.passphrase = os.getenv(env_passphrase)
        
        # Database credentials
        if os.getenv("DB_PASSWORD"):
            self.database.password = os.getenv("DB_PASSWORD")
        if os.getenv("DB_USERNAME"):
            self.database.username = os.getenv("DB_USERNAME")
        if os.getenv("DB_HOST"):
            self.database.host = os.getenv("DB_HOST")
    
    def create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            "exchanges": {
                "kraken": {
                    "api_key": "your_kraken_api_key",
                    "api_secret": "your_kraken_api_secret",
                    "sandbox": True,
                    "rate_limit": 10,
                    "timeout": 30
                }
            },
            "strategies": {
                "momentum_strategy": {
                    "enabled": True,
                    "parameters": {
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "ma_short": 10,
                        "ma_long": 20
                    },
                    "symbols": ["XBTUSD", "ETHUSD"],
                    "max_position_size": "0.1",
                    "stop_loss_percentage": 0.05,
                    "take_profit_percentage": 0.10
                }
            },
            "risk": {
                "max_portfolio_risk": 0.02,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.10,
                "position_sizing_method": "fixed_percentage",
                "emergency_stop_loss": 0.15
            },
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "trading_bot",
                "username": "trader",
                "password": "",
                "pool_size": 10
            },
            "ai": {
                "model_type": "lstm",
                "feature_window": 50,
                "prediction_horizon": 5,
                "retrain_frequency": 24,
                "min_training_data": 1000
            },
            "notifications": {
                "enabled": True,
                "channels": ["email"],
                "email_smtp_server": "smtp.gmail.com:587",
                "email_username": "",
                "email_password": "",
                "slack_webhook": "",
                "telegram_token": "",
                "telegram_chat_id": ""
            },
            "log_level": "INFO",
            "data_directory": "data",
            "backup_enabled": True,
            "backup_frequency": 24
        }
        
        # Ensure config directory exists
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"Created default configuration file at {self.config_path}")
        print("Please update the configuration with your API keys and preferences")
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            "exchanges": {name: asdict(config) for name, config in self.exchanges.items()},
            "strategies": {name: asdict(config) for name, config in self.strategies.items()},
            "risk": asdict(self.risk),
            "database": asdict(self.database),
            "ai": asdict(self.ai),
            "notifications": asdict(self.notifications),
            "log_level": self.log_level,
            "data_directory": self.data_directory,
            "backup_enabled": self.backup_enabled,
            "backup_frequency": self.backup_frequency
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for a specific exchange"""
        return self.exchanges.get(exchange_name)
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy"""
        return self.strategies.get(strategy_name)
    
    def add_exchange(self, exchange_config: ExchangeConfig):
        """Add a new exchange configuration"""
        self.exchanges[exchange_config.name] = exchange_config
    
    def add_strategy(self, strategy_config: StrategyConfig):
        """Add a new strategy configuration"""
        self.strategies[strategy_config.name] = strategy_config
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate exchanges
        if not self.exchanges:
            errors.append("No exchanges configured")
        
        for name, exchange in self.exchanges.items():
            if not exchange.api_key or exchange.api_key == "your_kraken_api_key":
                errors.append(f"Invalid API key for exchange {name}")
            if not exchange.api_secret or exchange.api_secret == "your_kraken_api_secret":
                errors.append(f"Invalid API secret for exchange {name}")
        
        # Validate strategies
        if not self.strategies:
            errors.append("No strategies configured")
        
        for name, strategy in self.strategies.items():
            if not strategy.symbols:
                errors.append(f"No symbols configured for strategy {name}")
            if strategy.max_position_size <= 0:
                errors.append(f"Invalid position size for strategy {name}")
        
        # Validate risk parameters
        if self.risk.max_portfolio_risk <= 0 or self.risk.max_portfolio_risk > 1:
            errors.append("Invalid max_portfolio_risk (should be between 0 and 1)")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Example usage
if __name__ == "__main__":
    config = Config()
    
    if config.validate_config():
        print("Configuration is valid!")
        print(f"Configured exchanges: {list(config.exchanges.keys())}")
        print(f"Configured strategies: {list(config.strategies.keys())}")
    else:
        print("Configuration validation failed!")
    
    # Example: Add a new exchange
    binance_config = ExchangeConfig(
        name="binance",
        api_key="your_binance_api_key",
        api_secret="your_binance_api_secret",
        sandbox=True
    )
    config.add_exchange(binance_config)
    config.save_config()