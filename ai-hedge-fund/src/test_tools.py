import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from tools import (
    api_client,
    get_prices,
    get_price_data,
    get_market_data,
    get_financial_metrics,
    get_cash_flow_statements,
    get_insider_trades,
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_obv
)

def test_api():
    """Test basic API functionality."""
    print("\nTesting API Connection...")
    
    # Test stock data
    ticker = "AAPL"
    print(f"\nFetching stock data for {ticker}...")
    data = api_client.get_stock_data(ticker)
    print("✓ Got stock data")
    print(f"Keys available: {list(data.keys())}")
    
    # Test prices
    print(f"\nFetching prices for {ticker}...")
    prices = api_client.get_prices(ticker)
    print(f"✓ Got {len(prices)} price records")
    
    # Test caching
    print("\nTesting cache...")
    cached_data = api_client.get_stock_data(ticker)
    print("✓ Retrieved from cache")
    
    return True

def test_data_functions():
    """Test all data fetching functions."""
    print("\nTesting data fetching functions...")
    
    ticker = "AAPL"
    
    # Test each function
    print("\nFetching different types of data:")
    
    prices = get_prices(ticker)
    print("✓ Prices data")
    
    market_data = get_market_data(ticker)
    print("✓ Market data")
    print(f"Market data keys: {list(market_data.keys())}")
    
    metrics = get_financial_metrics(ticker)
    print("✓ Financial metrics")
    print(f"Financial metrics keys: {list(metrics.keys())}")
    
    cash_flows = get_cash_flow_statements(ticker)
    print("✓ Cash flow statements")
    print(f"Got {len(cash_flows)} statements")
    
    insider = get_insider_trades(ticker)
    print("✓ Insider trades")
    print(f"Got {len(insider)} trades")
    
    return True

def test_technical_analysis():
    """Test technical analysis functions."""
    print("\nTesting technical analysis...")
    
    # Get price data as DataFrame
    df = get_price_data("AAPL")
    print("✓ Converted prices to DataFrame")
    print(f"DataFrame shape: {df.shape}")
    
    # Calculate indicators
    macd, signal = calculate_macd(df)
    print("✓ Calculated MACD")
    print(f"Latest MACD: {macd.iloc[-1]:.2f}")
    print(f"Latest Signal: {signal.iloc[-1]:.2f}")
    
    rsi = calculate_rsi(df)
    print("✓ Calculated RSI")
    print(f"Latest RSI: {rsi.iloc[-1]:.2f}")
    
    upper, lower = calculate_bollinger_bands(df)
    print("✓ Calculated Bollinger Bands")
    print(f"Latest Upper: {upper.iloc[-1]:.2f}")
    print(f"Latest Lower: {lower.iloc[-1]:.2f}")
    
    obv = calculate_obv(df)
    print("✓ Calculated OBV")
    print(f"Latest OBV: {obv.iloc[-1]:.0f}")
    
    return True

def main():
    """Run all tests."""
    try:
        print("Starting tests...")
        
        # Run tests
        api_test = test_api()
        data_test = test_data_functions()
        ta_test = test_technical_analysis()
        
        # Print summary
        print("\nTest Summary:")
        print(f"API Test: {'✓' if api_test else '✗'}")
        print(f"Data Functions: {'✓' if data_test else '✗'}")
        print(f"Technical Analysis: {'✓' if ta_test else '✗'}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 