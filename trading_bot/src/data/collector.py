# src/data/collector.py
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path
import logging
from decimal import Decimal

@dataclass
class OHLCV:
    """OHLCV data structure"""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str

@dataclass
class OrderBookSnapshot:
    """Order book snapshot data"""
    timestamp: datetime
    symbol: str
    bids: List[tuple[Decimal, Decimal]]  # (price, volume)
    asks: List[tuple[Decimal, Decimal]]  # (price, volume)

@dataclass
class Trade:
    """Trade data structure"""
    timestamp: datetime
    symbol: str
    price: Decimal
    volume: Decimal
    side: str  # 'buy' or 'sell'

class DataCollector:
    """Collect and store market data from exchanges"""
    
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config.data_directory) / "market_data.db"
        self.is_collecting = False
        self.collection_tasks = []
        
        # Create data directory
        Path(config.data_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing market data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # Order book snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    bids TEXT NOT NULL,  -- JSON string
                    asks TEXT NOT NULL   -- JSON string
                )
            """)
            
            # Trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    side TEXT NOT NULL
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp ON orderbook(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)")
            
            conn.commit()
    
    async def start_collection(self, symbols: List[str], intervals: List[str] = ["1m"]):
        """Start collecting data for specified symbols"""
        if self.is_collecting:
            self.logger.warning("Data collection already in progress")
            return
        
        self.is_collecting = True
        self.logger.info(f"Starting data collection for symbols: {symbols}")
        
        # Create collection tasks for each symbol and interval
        for symbol in symbols:
            for interval in intervals:
                task = asyncio.create_task(self._collect_ohlcv_data(symbol, interval))
                self.collection_tasks.append(task)
            
            # Collect order book data
            task = asyncio.create_task(self._collect_orderbook_data(symbol))
            self.collection_tasks.append(task)
        
        # Wait for all tasks to complete (they run indefinitely)
        try:
            await asyncio.gather(*self.collection_tasks)
        except asyncio.CancelledError:
            self.logger.info("Data collection stopped")
    
    async def stop_collection(self):
        """Stop data collection"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        self.logger.info("Stopping data collection...")
        
        # Cancel all collection tasks
        for task in self.collection_tasks:
            task.cancel()
        
        # Wait for tasks to finish cancelling
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        self.collection_tasks.clear()
        
        self.logger.info("Data collection stopped")
    
    async def _collect_ohlcv_data(self, symbol: str, interval: str = "1m"):
        """Collect OHLCV data for a symbol"""
        interval_seconds = self._interval_to_seconds(interval)
        
        while self.is_collecting:
            try:
                # Get OHLC data from exchange
                ohlc_data = await self.exchange.get_ohlc_data(symbol, interval=1)  # 1 minute
                
                if ohlc_data:
                    # Store the latest candle
                    latest_candle = ohlc_data[-1]
                    timestamp = datetime.fromtimestamp(latest_candle[0])
                    
                    ohlcv = OHLCV(
                        timestamp=timestamp,
                        open=Decimal(str(latest_candle[1])),
                        high=Decimal(str(latest_candle[2])),
                        low=Decimal(str(latest_candle[3])),
                        close=Decimal(str(latest_candle[4])),
                        volume=Decimal(str(latest_candle[6])),
                        symbol=symbol
                    )
                    
                    await self.store_ohlcv(ohlcv)
                    self.logger.debug(f"Stored OHLCV data for {symbol} at {timestamp}")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error collecting OHLCV data for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_orderbook_data(self, symbol: str, interval: int = 5):
        """Collect order book snapshots"""
        while self.is_collecting:
            try:
                orderbook_data = await self.exchange.get_orderbook(symbol, depth=20)
                
                if orderbook_data:
                    bids = [(Decimal(str(bid[0])), Decimal(str(bid[1]))) 
                           for bid in orderbook_data.get('bids', [])]
                    asks = [(Decimal(str(ask[0])), Decimal(str(ask[1]))) 
                           for ask in orderbook_data.get('asks', [])]
                    
                    snapshot = OrderBookSnapshot(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        bids=bids,
                        asks=asks
                    )
                    
                    await self.store_orderbook_snapshot(snapshot)
                    self.logger.debug(f"Stored order book snapshot for {symbol}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting order book data for {symbol}: {e}")
                await asyncio.sleep(5)
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds"""
        if interval.endswith('m'):
            return int(interval[:-1]) * 60
        elif interval.endswith('h'):
            return int(interval[:-1]) * 3600
        elif interval.endswith('d'):
            return int(interval[:-1]) * 86400
        else:
            return 60  # Default to 1 minute
    
    async def store_ohlcv(self, ohlcv: OHLCV):
        """Store OHLCV data in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (timestamp, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(ohlcv.timestamp.timestamp()),
                    ohlcv.symbol,
                    float(ohlcv.open),
                    float(ohlcv.high),
                    float(ohlcv.low),
                    float(ohlcv.close),
                    float(ohlcv.volume)
                ))
                conn.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Error storing OHLCV data: {e}")
    
    async def store_orderbook_snapshot(self, snapshot: OrderBookSnapshot):
        """Store order book snapshot in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                bids_json = json.dumps([[float(price), float(volume)] for price, volume in snapshot.bids])
                asks_json = json.dumps([[float(price), float(volume)] for price, volume in snapshot.asks])
                
                cursor.execute("""
                    INSERT INTO orderbook (timestamp, symbol, bids, asks)
                    VALUES (?, ?, ?, ?)
                """, (
                    int(snapshot.timestamp.timestamp()),
                    snapshot.symbol,
                    bids_json,
                    asks_json
                ))
                conn.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Error storing order book snapshot: {e}")
    
    def get_ohlcv_data(self, symbol: str, start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve OHLCV data as pandas DataFrame"""
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Get the latest price for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT close FROM ohlcv 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            return Decimal(str(result[0])) if result else None