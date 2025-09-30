# src/exchanges/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime

class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get order book data"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: Decimal, 
                         price: Optional[Decimal] = None, order_type: str = "market") -> Dict[str, Any]:
        """Place a trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_ohlc_data(self, symbol: str, interval: int = 1, since: Optional[int] = None) -> List[List]:
        """Get OHLC (candlestick) data"""
        pass
    
    @abstractmethod
    async def get_open_orders(self) -> Dict[str, Any]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_closed_orders(self, start: Optional[int] = None, end: Optional[int] = None) -> Dict[str, Any]:
        """Get closed orders"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the connection"""
        pass