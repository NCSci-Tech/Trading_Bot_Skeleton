# Complete Multi-Language Trading Bot Setup Guide

## Overview
This guide walks you through setting up a complete trading bot system with:
- **Python**: Core bot logic, strategies, data processing
- **Rust**: High-performance technical indicators and backtesting
- **Solidity**: DeFi integration and on-chain strategies

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space (SSD recommended)
- **Network**: Stable internet connection

### Software Dependencies
- Python 3.9+
- Rust 1.70+
- Node.js 16+ (for Solidity development)
- PostgreSQL 13+ (optional, SQLite works too)
- Git

## Installation Steps

### 1. System Setup

```bash
# Update system packages (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git vim

# Install Python development tools
sudo apt install -y python3-dev python3-pip python3-venv

# Install PostgreSQL (optional)
sudo apt install -y postgresql postgresql-contrib
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url> trading-bot
cd trading-bot

# Create directory structure
mkdir -p {src/{core,exchanges,strategies,data,risk,ai,utils,defi},rust_modules/{indicators,orderbook,backtesting},contracts/{interfaces,strategies,utils},tests,config,data,logs,wheels}

# Set up Python virtual environment
python3 -m venv trading_bot_env
source trading_bot_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Python Environment Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install TA-Lib (technical analysis library)
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows (use pre-compiled wheel):
pip install --find-links https://wheels.scipy.org/ TA-Lib
```

### 4. Rust Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version

# Install maturin for Python-Rust bindings
pip install maturin

# Build Rust modules
python build_rust.py --clean --release

# Verify Rust modules work
python -c "import indicators, orderbook, backtesting; print(' Rust modules loaded successfully')"
```

### 5. Database Setup

#### Option A: PostgreSQL (Recommended for Production)

```bash
# Create database and user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE trading_bot;
CREATE USER trader WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trader;
\q
```

#### Option B: SQLite (Development/Testing)

```bash
# No setup needed - will create automatically
echo "Using SQLite - no additional setup required"
```

### 6. Configuration Setup

```bash
# Create configuration directory
mkdir -p config

# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration with your settings
vim config/config.yaml
```

**Example config/config.yaml:**

```yaml
exchanges:
  kraken:
    api_key: "your_kraken_api_key"
    api_secret: "your_kraken_api_secret"
    sandbox: true
    rate_limit: 10
    timeout: 30

strategies:
  momentum_strategy:
    enabled: true
    parameters:
      rsi_period: 14
      rsi_oversold: 30
      rsi_overbought: 70
      ma_short: 10
      ma_long: 20
      volume_threshold: 1.2
    symbols: ["XBTUSD", "ETHUSD"]
    max_position_size: "0.1"
    stop_loss_percentage: 0.05
    take_profit_percentage: 0.10

risk:
  max_portfolio_risk: 0.02
  max_daily_loss: 0.05
  max_drawdown: 0.10
  position_sizing_method: "fixed_percentage"
  emergency_stop_loss: 0.15

database:
  type: "sqlite"  # or "postgresql"
  host: "localhost"
  port: 5432
  database: "trading_bot"
  username: "trader"
  password: "your_secure_password"

ai:
  model_type: "lstm"
  feature_window: 50
  prediction_horizon: 5
  retrain_frequency: 24

notifications:
  enabled: true
  channels: ["email"]
  email_smtp_server: "smtp.gmail.com:587"
  email_username: "your_email@gmail.com"
  email_password: "your_app_password"

log_level: "INFO"
data_directory: "data"
```

### 7. Environment Variables Setup

```bash
# Create .env file for sensitive data
cat > .env << EOF
# Exchange API Keys
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret

# Database
DB_PASSWORD=your_secure_password
DB_USERNAME=trader
DB_HOST=localhost

# Notifications
EMAIL_PASSWORD=your_app_password
SLACK_WEBHOOK=your_slack_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# For DeFi (optional)
PRIVATE_KEY=your_ethereum_private_key
INFURA_PROJECT_ID=your_infura_project_id
EOF

# Secure the .env file
chmod 600 .env
```

## Testing the Setup

### 1. Test Python Components

```bash
# Activate virtual environment
source trading_bot_env/bin/activate

# Test configuration loading
python -c "
from src.core.config import Config
config = Config()
print('Configuration loaded successfully')
print(f'Exchanges: {list(config.exchanges.keys())}')
print(f'Strategies: {list(config.strategies.keys())}')
"

# Test data processor
python -c "
from src.data.processor import DataProcessor
import numpy as np
processor = DataProcessor(None)
prices = np.random.randn(100) + 50000
sma = processor.calculate_technical_indicators(
    __import__('pandas').DataFrame({'open': prices, 'high': prices*1.01, 'low': prices*0.99, 'close': prices, 'volume': np.ones(100)*1000})
)
print('Technical indicators calculated successfully')
"
```

### 2. Test Rust Components

```bash
# Test technical indicators
python -c "
from src.utils.rust_bridge import RustIndicators
import numpy as np
indicators = RustIndicators()
prices = np.array([50000.0, 50100.0, 50200.0, 50150.0, 50300.0])
sma = indicators.calculate_sma(prices, 3)
print(f'Rust SMA: {sma}')
"

# Test orderbook
python -c "
from src.utils.rust_bridge import RustOrderBook
ob = RustOrderBook('BTCUSD')
ob.update_bid(50000, 1.5)
ob.update_ask(50100, 2.0)
print(f'Rust OrderBook spread: {ob.get_spread()}')
"

# Test backtesting
python -c "
from src.utils.rust_bridge import RustBacktester
bt = RustBacktester(10000, 0.001, 0.0001)
success = bt.execute_trade('2023-01-01T00:00:00Z', 'BTCUSD', 'buy', 50000, 0.1)
print(f'Rust Backtester trade: {success}')
"
```

### 3. Performance Benchmark

```bash
# Run performance comparison
python -c "
from src.utils.rust_bridge import benchmark_indicators
import numpy as np
np.random.seed(42)
prices = 50000 + np.cumsum(np.random.randn(10000) * 100)
results = benchmark_indicators(prices, iterations=100)
for metric, speedup in results.items():
    print(f'{metric}: {speedup:.2f}x speedup with Rust')
"
```

## Running the Trading Bot

### 1. Dry Run (Paper Trading)

```bash
# Ensure sandbox mode is enabled in config
# Set all exchange configurations to sandbox: true

# Start the bot in dry run mode
python -m src.core.bot --dry-run

# Or run with specific log level
python -m src.core.bot --log-level DEBUG
```

### 2. Live Trading (Production)

**WARNING**: Only run live trading after thorough testing!

```bash
# Update configuration for live trading
# Set sandbox: false in exchange configurations
# Use small position sizes initially

# Start live trading
python -m src.core.bot --live

# Monitor with logs
tail -f logs/trading_bot.log
```

### 3. Background Process (Production Deployment)

```bash
# Create systemd service file
sudo tee /etc/systemd/system/trading-bot.service > /dev/null << EOF
[Unit]
Description=Cryptocurrency Trading Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=$(whoami)
ExecStart=$(pwd)/trading_bot_env/bin/python -m src.core.bot --live
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/trading_bot_env/bin

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service

# Check status
sudo systemctl status trading-bot.service

# View logs
sudo journalctl -u trading-bot.service -f
```

## Solidity/DeFi Setup (Optional)

### 1. Node.js and Hardhat Setup

```bash
# Install Node.js dependencies
npm init -y
npm install --save-dev hardhat @nomiclabs/hardhat-ethers ethers
npm install @openzeppelin/contracts

# Initialize Hardhat
npx hardhat init

# Install additional dependencies
npm install @uniswap/v3-periphery @aave/protocol-v2
```

### 2. Smart Contract Configuration

```javascript
// hardhat.config.js
require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {},
    goerli: {
      url: `https://goerli.infura.io/v3/${process.env.INFURA_PROJECT_ID}`,
      accounts: [process.env.PRIVATE_KEY]
    },
    mainnet: {
      url: `https://mainnet.infura.io/v3/${process.env.INFURA_PROJECT_ID}`,
      accounts: [process.env.PRIVATE_KEY]
    }
  }
};
```

### 3. Deploy Smart Contracts

```bash
# Compile contracts
npx hardhat compile

# Deploy to testnet
npx hardhat run scripts/deploy.js --network goerli

# Verify on Etherscan
npx hardhat verify --network goerli DEPLOYED_CONTRACT_ADDRESS "constructor_arg1" "constructor_arg2"
```

## Monitoring and Maintenance

### 1. Log Monitoring

```bash
# Real-time log monitoring
tail -f logs/trading_bot.log

# Search for errors
grep -i error logs/trading_bot.log

# Performance monitoring
grep -i "performance\|pnl\|profit" logs/trading_bot.log | tail -20
```

### 2. Database Maintenance

```bash
# PostgreSQL maintenance
sudo -u postgres psql trading_bot -c "VACUUM ANALYZE;"

# SQLite maintenance
sqlite3 data/market_data.db "VACUUM;"

# Backup database
pg_dump trading_bot > backups/trading_bot_$(date +%Y%m%d_%H%M%S).sql
```

### 3. Performance Optimization

```bash
# Update Rust modules for performance
cd rust_modules
export RUSTFLAGS="-C target-cpu=native"
python ../build_rust.py --clean --release

# Profile Python code
python -m cProfile -o profile.stats -m src.core.bot --dry-run
```

## Troubleshooting

### Common Issues

1. **Rust modules not loading**
   ```bash
   # Rebuild with verbose output
   python build_rust.py --clean --release --verbose
   
   # Check Python can find modules
   python -c "import sys; print(sys.path)"
   ```

2. **API connection failures**
   ```bash
   # Test API credentials
   python -c "
   from src.exchanges.kraken import KrakenExchange
   import asyncio
   async def test():
       kraken = KrakenExchange('api_key', 'api_secret', True)
       connected = await kraken.connect()
       print(f'Connected: {connected}')
       await kraken.close()
   asyncio.run(test())
   "
   ```

3. **Database connection issues**
   ```bash
   # Test PostgreSQL connection
   psql -h localhost -U trader -d trading_bot -c "SELECT version();"
   
   # Check SQLite permissions
   ls -la data/market_data.db
   ```

4. **Memory issues**
   ```bash
   # Monitor memory usage
   ps aux | grep python
   
   # Limit memory usage in config
   # Add to config: max_memory_gb: 4
   ```

### Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check inline comments and docstrings
- **Community**: Join trading bot communities for support
- **Logs**: Always check logs first for error details

## Security Best Practices

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables or encrypted configs
   - Regularly rotate API keys
   - Use API keys with minimal required permissions

2. **System Security**
   - Keep system and dependencies updated
   - Use firewall to restrict network access
   - Run bot with minimal user privileges
   - Regular security audits

3. **Risk Management**
   - Start with small position sizes
   - Set strict stop-loss limits
   - Monitor positions actively
   - Have emergency stop procedures

## Production Deployment Checklist

- [ ] All tests pass
- [ ] Configuration validated
- [ ] API keys secured
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backups configured
- [ ] Emergency procedures documented
- [ ] Risk limits set appropriately
- [ ] Performance benchmarks established
- [ ] Alert systems configured

## Next Steps

1. **Backtesting**: Run extensive backtests on historical data
2. **Paper Trading**: Test with virtual money for at least 1 month
3. **Gradual Scaling**: Start with small amounts and scale up slowly
4. **Strategy Development**: Develop and test custom strategies
5. **Performance Analysis**: Regular analysis of bot performance
6. **Risk Assessment**: Continuous risk monitoring and adjustment

---

**Remember**: Trading involves significant risk. Never trade with money you can't afford to lose, and always thoroughly test any trading system before using it with real money.