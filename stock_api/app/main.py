from datetime import datetime, timedelta, date
from typing import List, Optional, Dict
from enum import Enum
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app import models, database
from app.config import settings
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Interval(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

app = FastAPI(
    title="Stock Market API",
    description="A FastAPI service providing market data for trading agents",
    version="1.0.0",
)

# Initialize database
models.Base.metadata.create_all(bind=database.engine)

class PriceResponse(BaseModel):
    open: float = Field(..., description="Opening price")
    close: float = Field(..., description="Closing price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    volume: int = Field(..., description="Trading volume")
    time: str = Field(..., description="Timestamp")
    time_milliseconds: int = Field(..., description="Unix timestamp in milliseconds")

class StockPricesResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    interval: str = Field(..., description="Time interval")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    prices: List[PriceResponse]
    warning: Optional[str] = Field(None, description="Warning message about data limitations")

class StockData(BaseModel):
    ticker: str
    income_statements: List[dict]
    balance_sheets: List[dict]
    cash_flow_statements: List[dict]
    financial_ratios: dict
    insider_transactions: List[dict]
    analyst_grades: List[dict]
    insider_sentiment: Optional[dict] = None
    market_data: Optional[dict] = None
    options_data: Optional[dict] = None

def _process_minute_data(df: pd.DataFrame) -> List[Dict]:
    """Process minute-level price data."""
    result = []
    for date in pd.unique(df.index.date):
        day_data = df[df.index.date == date]
        if day_data.empty:
            continue
            
        # Filter out zero volume records and daily records at midnight
        day_data = day_data[
            ((day_data['volume'] > 0) | 
            (day_data.index.hour < 9) | 
            (day_data.index.hour > 16)) &
            ~((day_data.index.hour == 0) & (day_data.index.minute == 0))
        ]
        
        # Convert to list of price records
        result.extend([
            {
                "timestamp": timestamp.to_pydatetime(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": int(row["volume"])
            }
            for timestamp, row in day_data.sort_index().iterrows()
        ])
    
    return result

def _process_hourly_data(df: pd.DataFrame, interval_multiplier: int) -> List[Dict]:
    """Process hourly price data within market hours (9:30 AM - 4:00 PM)."""
    result = []
    for date in pd.unique(df.index.date):
        day_data = df[df.index.date == date]
        if day_data.empty:
            continue
            
        # Process each hour from market open to close
        current_time = pd.Timestamp.combine(date, pd.Timestamp("9:30").time())
        while current_time.time() <= pd.Timestamp("16:00").time():
            next_time = current_time + pd.Timedelta(hours=interval_multiplier)
            hour_data = day_data[
                (day_data.index >= current_time) &
                (day_data.index < next_time)
            ]
            
            if not hour_data.empty:
                result.append({
                    "timestamp": current_time.to_pydatetime(),
                    "open": hour_data.iloc[0]["open"],
                    "high": hour_data["high"].max(),
                    "low": hour_data["low"].min(),
                    "close": hour_data.iloc[-1]["close"],
                    "volume": int(hour_data["volume"].sum())
                })
            
            current_time = next_time
    
    return result

def _has_intraday_data(df: pd.DataFrame) -> bool:
    """Check if the dataframe contains intraday data."""
    return len(df.index) > 1 and any(
        ts1.date() == ts2.date() and (ts1.hour != ts2.hour or ts1.minute != ts2.minute)
        for ts1, ts2 in zip(df.index[:-1], df.index[1:])
    )

def aggregate_prices(prices: List[models.StockPrice], interval: Interval, interval_multiplier: int) -> List[Dict]:
    """Aggregate prices based on the specified interval."""
    if not prices:
        return []
    
    # Convert to DataFrame and set timestamp index
    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(price.timestamp),
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume
        }
        for price in prices
    ]).set_index("timestamp")
    
    df.index = pd.to_datetime(df.index)
    
    # Handle intraday intervals
    if interval in [Interval.MINUTE, Interval.HOUR]:
        if not _has_intraday_data(df):
            return []
        
        if df.empty:
            return []
        
        return (_process_minute_data(df) if interval == Interval.MINUTE 
                else _process_hourly_data(df, interval_multiplier))
    
    # Handle daily and longer intervals
    rule = {
        Interval.DAY: f"{interval_multiplier}D",
        Interval.WEEK: f"{interval_multiplier}W",
        Interval.MONTH: f"{interval_multiplier}M"
    }[interval]
    
    # Aggregate data using resampling
    agg_df = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    return [
        {
            "timestamp": timestamp.to_pydatetime(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": int(row["volume"])
        }
        for timestamp, row in agg_df.iterrows()
    ]

@app.get("/prices", response_model=StockPricesResponse)
async def get_stock_prices(
    ticker: str = Query(..., description="Stock ticker symbol"),
    interval: Interval = Query(Interval.DAY, description="Time interval (minute/hour data only available for last 7 days)"),
    interval_multiplier: int = Query(1, gt=0, le=60),
    start_date: Optional[str] = Query(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date (optional)"
    ),
    end_date: Optional[str] = Query(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date (optional)"
    ),
    limit: int = Query(5000, gt=0, le=5000),
    db: Session = Depends(database.get_db),
):
    # Get stock from database
    stock = db.query(models.Stock).filter(models.Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found")
    
    warning_message = None
    
    # Handle end date
    try:
        end = (datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
               if end_date else datetime.now())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
    
    # Handle start date and 7-day limit for intraday data
    if interval in [Interval.MINUTE, Interval.HOUR]:
        seven_days_ago = (datetime.now() - timedelta(days=7)
                         ).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if start_date:
            user_start = (datetime.strptime(start_date, "%Y-%m-%d")
                         .replace(hour=0, minute=0, second=0, microsecond=0))
            
            if user_start < seven_days_ago:
                warning_message = f"{interval.value} data is only available for the last 7 days. Data will be limited to this period."
                start = seven_days_ago
            else:
                start = user_start
        else:
            start = seven_days_ago
    else:
        start = (datetime.strptime(start_date, "%Y-%m-%d")
                .replace(hour=0, minute=0, second=0, microsecond=0) if start_date else None)
    
    # Validate date range
    if start and end and end < start:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    
    # Query price data
    query = db.query(models.StockPrice).filter(models.StockPrice.stock_id == stock.id)
    if start:
        query = query.filter(models.StockPrice.timestamp >= start)
    if end:
        query = query.filter(models.StockPrice.timestamp <= end)
    
    prices = query.order_by(models.StockPrice.timestamp.asc()).all()
    if not prices:
        raise HTTPException(status_code=404, detail=f"No prices found for '{ticker}' in the specified date range")
    
    # Aggregate and limit results
    aggregated_prices = aggregate_prices(prices, interval, interval_multiplier)[:limit]
    if not aggregated_prices:
        raise HTTPException(
            status_code=404, 
            detail=f"No data available for the specified interval and date range"
        )
    
    # Get actual date range from the data
    actual_start = min(price["timestamp"] for price in aggregated_prices).strftime("%Y-%m-%d")
    actual_end = max(price["timestamp"] for price in aggregated_prices).strftime("%Y-%m-%d")
    
    return StockPricesResponse(
        ticker=ticker.upper(),
        interval=f"{interval_multiplier}-{interval.value}",
        start_date=actual_start,
        end_date=actual_end,
        warning=warning_message,
        prices=[
            PriceResponse(
                open=price["open"],
                close=price["close"],
                high=price["high"],
                low=price["low"],
                volume=price["volume"],
                time=price["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                time_milliseconds=int(price["timestamp"].timestamp() * 1000)
            )
            for price in aggregated_prices
        ]
    )

@app.get("/stock/{ticker}", response_model=StockData)
async def get_stock_data(
    ticker: str,
    start_date: str = Query("2021-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d"), pattern=r"^\d{4}-\d{2}-\d{2}$"),
    db: Session = Depends(database.get_db),
):
    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end < start:
            raise HTTPException(status_code=400, detail="end_date must be after start_date")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Get stock
    stock = db.query(models.Stock).filter(models.Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found")
    
    # Get all data from database with date filtering
    income_statements = [
        {
            "date": stmt.calendar_date.strftime('%Y-%m-%d'),
            "period": stmt.period.value.lower(),
            "total_revenue": stmt.total_revenue,
            "net_income": stmt.net_income,
            "operating_income": stmt.operating_income,
            "gross_profit": stmt.gross_profit,
            "operating_expense": stmt.operating_expense,
            "ebit": stmt.ebit,
            "net_margin": stmt.net_margin,
            "operating_margin": stmt.operating_margin
        }
        for stmt in db.query(models.IncomeStatement)
        .filter(models.IncomeStatement.stock_id == stock.id)
        .filter(models.IncomeStatement.calendar_date >= start)
        .filter(models.IncomeStatement.calendar_date <= end)
        .order_by(models.IncomeStatement.calendar_date.desc())
        .all()
    ]

    balance_sheets = [
        {
            "date": stmt.calendar_date.strftime('%Y-%m-%d'),
            "period": stmt.period.value.lower(),
            "total_assets": stmt.total_assets,
            "total_liabilities": stmt.total_liabilities,
            "stockholders_equity": stmt.stockholders_equity,
            "cash_and_equivalents": stmt.cash_and_equivalents,
            "total_debt": stmt.total_debt,
            "current_assets": stmt.current_assets,
            "current_liabilities": stmt.current_liabilities
        }
        for stmt in db.query(models.BalanceSheet)
        .filter(models.BalanceSheet.stock_id == stock.id)
        .filter(models.BalanceSheet.calendar_date >= start)
        .filter(models.BalanceSheet.calendar_date <= end)
        .order_by(models.BalanceSheet.calendar_date.desc())
        .all()
    ]

    cash_flow_statements = [
        {
            "date": stmt.calendar_date.strftime('%Y-%m-%d'),
            "period": stmt.period.value.lower(),
            "operating_cash_flow": stmt.operating_cash_flow,
            "free_cash_flow": stmt.free_cash_flow
        }
        for stmt in db.query(models.CashFlowStatement)
        .filter(models.CashFlowStatement.stock_id == stock.id)
        .filter(models.CashFlowStatement.calendar_date >= start)
        .filter(models.CashFlowStatement.calendar_date <= end)
        .order_by(models.CashFlowStatement.calendar_date.desc())
        .all()
    ]

    insider_transactions = [
        {
            "date": tx.date.strftime('%Y-%m-%d'),
            "insider_name": tx.insider_name,
            "type": tx.transaction_type,
            "shares": tx.shares,
            "value": tx.value,
            "price": tx.transaction_price
        }
        for tx in db.query(models.InsiderTransaction)
        .filter(models.InsiderTransaction.stock_id == stock.id)
        .filter(models.InsiderTransaction.date >= start)
        .filter(models.InsiderTransaction.date <= end)
        .order_by(models.InsiderTransaction.date.desc())
        .all()
    ]

    analyst_grades = [
        {
            "date": grade.grade_date.strftime('%Y-%m-%d'),
            "firm": grade.firm,
            "action": grade.action.value,
            "from_grade": grade.from_grade,
            "to_grade": grade.to_grade
        }
        for grade in db.query(models.AnalystGrade)
        .filter(models.AnalystGrade.stock_id == stock.id)
        .filter(models.AnalystGrade.grade_date >= start)
        .filter(models.AnalystGrade.grade_date <= end)
        .order_by(models.AnalystGrade.grade_date.desc())
        .all()
    ]

    # Get latest data points for single-record tables
    insider_sentiment = db.query(models.InsiderSentiment).filter(
        models.InsiderSentiment.stock_id == stock.id,
        models.InsiderSentiment.date >= start,
        models.InsiderSentiment.date <= end
    ).order_by(models.InsiderSentiment.date.desc()).first()

    market_data = db.query(models.MarketData).filter(
        models.MarketData.stock_id == stock.id,
        models.MarketData.date >= start,
        models.MarketData.date <= end
    ).order_by(models.MarketData.date.desc()).first()

    options_data = db.query(models.OptionsData).filter(
        models.OptionsData.stock_id == stock.id,
        models.OptionsData.date >= start,
        models.OptionsData.date <= end
    ).order_by(models.OptionsData.date.desc()).first()

    financial_ratios = db.query(models.FinancialRatios).filter(
        models.FinancialRatios.stock_id == stock.id,
        models.FinancialRatios.date >= start,
        models.FinancialRatios.date <= end
    ).order_by(models.FinancialRatios.date.desc()).first()

    return StockData(
        ticker=ticker.upper(),
        income_statements=income_statements,
        balance_sheets=balance_sheets,
        cash_flow_statements=cash_flow_statements,
        financial_ratios={
            "date": financial_ratios.date.strftime('%Y-%m-%d'),
            "pe_ratio": financial_ratios.pe_ratio,
            "pb_ratio": financial_ratios.pb_ratio,
            "ps_ratio": financial_ratios.ps_ratio,
            "debt_to_equity": financial_ratios.debt_to_equity,
            "current_ratio": financial_ratios.current_ratio,
            "return_on_equity": financial_ratios.return_on_equity,
            "profit_margins": financial_ratios.profit_margins,
            "operating_margins": financial_ratios.operating_margins,
            "earnings_growth": financial_ratios.earnings_growth,
            "revenue_growth": financial_ratios.revenue_growth,
            "free_cash_flow": financial_ratios.free_cash_flow,
            "free_cash_flow_per_share": financial_ratios.free_cash_flow_per_share,
            "earnings_per_share": financial_ratios.earnings_per_share,
            "beta": financial_ratios.beta
        } if financial_ratios else None,
        insider_transactions=insider_transactions,
        analyst_grades=analyst_grades,
        insider_sentiment={
            "date": insider_sentiment.date.strftime('%Y-%m-%d'),
            "net_shares": insider_sentiment.net_shares,
            "net_shares_pct": insider_sentiment.net_shares_pct,
            "buy_shares_pct": insider_sentiment.buy_shares_pct,
            "sell_shares_pct": insider_sentiment.sell_shares_pct,
            "total_transactions": insider_sentiment.total_transactions
        } if insider_sentiment else None,
        market_data={
            "date": market_data.date.strftime('%Y-%m-%d'),
            "beta": market_data.beta,
            "sector": market_data.sector,
            "industry": market_data.industry,
            "float_shares": market_data.float_shares,
            "shares_outstanding": market_data.shares_outstanding,
            "market_cap": market_data.market_cap
        } if market_data else None,
        options_data={
            "date": options_data.date.strftime('%Y-%m-%d'),
            "implied_volatility": options_data.implied_volatility,
            "put_call_ratio": options_data.put_call_ratio,
            "volume": options_data.options_volume
        } if options_data else None
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        reload=settings.debug_mode,
        port=settings.port,
    ) 