# src/exchanges/kraken.py
import hashlib
import hmac
import base64
import time
import json
import aiohttp
from urllib.parse import urlencode
from typing import Dict, List, Optional, Any
from decimal import Decimal
from .base import ExchangeInterface

class KrakenExchange(ExchangeInterface):
    """Kraken Pro API implementation"""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = "https://api.kraken.com" if not sandbox else "https://demo-futures.kraken.com"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self) -> bool:
        """Initialize aiohttp session"""
        try:
            self.session = aiohttp.ClientSession()
            # Test connection with server time
            async with self.session.get(f"{self.base_url}/0/public/Time") as response:
                if response.status == 200:
                    return True
                return False
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def _generate_signature(self, endpoint: str, data: str, nonce: str) -> str:
        """Generate API signature for authenticated requests"""
        postdata = f"nonce={nonce}&{data}"
        encoded = (nonce + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self.api_secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    async def _make_request(self, endpoint: str, data: Optional[Dict] = None, 
                           method: str = "GET", private: bool = False) -> Dict[str, Any]:
        """Make API request with proper authentication"""
        if not self.session:
            raise Exception("Not connected. Call connect() first.")
        
        url = f"{self.base_url}{endpoint}"
        headers = {"User-Agent": "TradingBot/1.0"}
        
        if private:
            nonce = str(int(time.time() * 1000))
            if data is None:
                data = {}
            data["nonce"] = nonce
            
            if method == "POST":
                postdata = urlencode(data)
                headers["API-Key"] = self.api_key
                headers["API-Sign"] = self._generate_signature(endpoint, postdata, nonce)
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            
        try:
            if method == "GET":
                async with self.session.get(url, headers=headers, params=data) as response:
                    result = await response.json()
            else:  # POST
                async with self.session.post(url, headers=headers, data=data) as response:
                    result = await response.json()
            
            if result.get("error"):
                raise Exception(f"API Error: {result['error']}")
            
            return result.get("result", {})
            
        except Exception as e:
            raise Exception(f"Request failed: {e}")
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance"""
        result = await self._make_request("/0/private/Balance", method="POST", private=True)
        return {asset: Decimal(balance) for asset, balance in result.items()}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data"""
        result = await self._make_request("/0/public/Ticker", {"pair": symbol})
        return result.get(symbol, {})
    
    async def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get order book data"""
        result = await self._make_request("/0/public/Depth", {"pair": symbol, "count": depth})
        return result.get(symbol, {})
    
    async def get_ohlc_data(self, symbol: str, interval: int = 1, since: Optional[int] = None) -> List[List]:
        """Get OHLC (candlestick) data"""
        params = {"pair": symbol, "interval": interval}
        if since:
            params["since"] = since
        
        result = await self._make_request("/0/public/OHLC", params)
        return result.get(symbol, [])
    
    async def place_order(self, symbol: str, side: str, amount: Decimal, 
                         price: Optional[Decimal] = None, order_type: str = "market") -> Dict[str, Any]:
        """Place a trading order"""
        order_data = {
            "pair": symbol,
            "type": side.lower(),  # buy or sell
            "ordertype": order_type.lower(),
            "volume": str(amount)
        }
        
        if price and order_type.lower() != "market":
            order_data["price"] = str(price)
        
        result = await self._make_request("/0/private/AddOrder", order_data, method="POST", private=True)
        return result
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        result = await self._make_request("/0/private/CancelOrder", 
                                        {"txid": order_id}, method="POST", private=True)
        return "count" in result and result["count"] > 0
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        result = await self._make_request("/0/private/QueryOrders", 
                                        {"txid": order_id}, method="POST", private=True)
        return result.get(order_id, {})
    
    async def get_open_orders(self) -> Dict[str, Any]:
        """Get all open orders"""
        return await self._make_request("/0/private/OpenOrders", method="POST", private=True)
    
    async def get_closed_orders(self, start: Optional[int] = None, end: Optional[int] = None) -> Dict[str, Any]:
        """Get closed orders"""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._make_request("/0/private/ClosedOrders", params, method="POST", private=True)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Example usage and testing
async def test_kraken_connection():
    """Test Kraken API connection and basic functionality"""
    # You'll need to set your actual API credentials
    API_KEY = "your_kraken_api_key"
    API_SECRET = "your_kraken_api_secret"
    
    kraken = KrakenExchange(API_KEY, API_SECRET, sandbox=True)
    
    try:
        # Test connection
        connected = await kraken.connect()
        print(f"Connection successful: {connected}")
        
        if connected:
            # Test public endpoints
            ticker = await kraken.get_ticker("XBTUSD")
            print(f"BTC/USD Ticker: {ticker}")
            
            orderbook = await kraken.get_orderbook("XBTUSD", 5)
            print(f"Order book depth: {len(orderbook.get('bids', []))}")
            
            # Test private endpoints (requires valid credentials)
            # balance = await kraken.get_balance()
            # print(f"Account balance: {balance}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await kraken.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_kraken_connection())