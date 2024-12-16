import os
from typing import Dict, Any, List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta

class StockAPI:
    """Client for interacting with our Stock API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self._cache = {}  # Cache for stock data
        
    def get_stock_data(
        self,
        ticker: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Dict:
        """Get comprehensive stock data including financials, sentiment, and market data."""
        cache_key = f"stock_{ticker}"
        
        if use_cache and not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
            
        url = f"{self.base_url}/stock/{ticker}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if use_cache:
            self._cache[cache_key] = data
            
        return data
    
    def get_prices(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> List[Dict]:
        """Get historical stock prices using API defaults (daily intervals)."""
        cache_key = f"prices_{ticker}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
            
        url = f"{self.base_url}/prices"
        params = {"ticker": ticker}  # All other parameters have API defaults
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        prices = response.json()["prices"]
        
        if use_cache:
            self._cache[cache_key] = prices
            
        return prices
    
    def clear_cache(self, ticker: Optional[str] = None):
        """Clear the cache for a specific ticker or all tickers."""
        if ticker:
            keys_to_remove = [k for k in self._cache.keys() if ticker in k]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

# Initialize global API client
api_client = StockAPI()

def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """Get financial metrics from our API."""
    data = api_client.get_stock_data(ticker)
    return data.get("financial_ratios", {})

def get_cash_flow_statements(ticker: str) -> List[Dict[str, Any]]:
    """Get cash flow statements from our API."""
    data = api_client.get_stock_data(ticker)
    return data.get("cash_flow_statements", [])

def get_insider_trades(ticker: str) -> List[Dict[str, Any]]:
    """Get insider trades from our API."""
    data = api_client.get_stock_data(ticker)
    return data.get("insider_transactions", [])

def get_market_data(ticker: str) -> Dict[str, Any]:
    """Get market data from our API."""
    data = api_client.get_stock_data(ticker)
    return data.get("market_data", {})

def get_prices(ticker: str) -> List[Dict[str, Any]]:
    """Get price data from our API."""
    return api_client.get_prices(ticker)

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.set_index("timestamp", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker: str) -> pd.DataFrame:
    """Get price data as a DataFrame."""
    prices = get_prices(ticker)
    return prices_to_df(prices)

# Technical Analysis Functions
def calculate_confidence_level(signals: dict) -> float:
    """Calculate confidence level based on the difference between SMAs."""
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line."""
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=prices_df.index)

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Calculate intrinsic value using the DCF method.
    Returns the intrinsic value in billions.
    """
    # Estimate future cash flows
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate present values
    present_values = [
        cf / (1 + discount_rate) ** (i + 1)
        for i, cf in enumerate(cash_flows)
    ]

    # Calculate terminal value
    terminal_value = (
        cash_flows[-1] * (1 + terminal_growth_rate) /
        (discount_rate - terminal_growth_rate)
    )
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up present values
    intrinsic_value = sum(present_values) + terminal_present_value

    # Convert to billions
    return intrinsic_value / 1e9
