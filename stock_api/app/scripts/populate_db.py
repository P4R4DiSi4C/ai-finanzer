import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
from functools import wraps
import time
import random

from app.database import SessionLocal, engine
from app.models import *

def retry_on_exception(retries=3, delay_base=1):
    """Decorator to retry a function on exception with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:  # Last attempt
                        raise
                    wait = (delay_base * (2 ** i) + 
                           random.uniform(0, 0.1 * delay_base * (2 ** i)))
                    print(f"Error in {func.__name__}: {str(e)}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
            return None
        return wrapper
    return decorator

def fetch_price_data(yf_ticker, stock_id, db, start_date, end_date):
    """Fetch price data with different granularity based on time period."""
    try:
        # Calculate the boundary for minute data (7 days ago)
        minute_data_start = end_date - timedelta(days=7)
        
        # Check existing data
        existing_data = db.query(StockPrice).filter(
            StockPrice.stock_id == stock_id,
            StockPrice.timestamp >= start_date,
            StockPrice.timestamp <= end_date
        ).order_by(StockPrice.timestamp.desc()).first()
        
        last_timestamp = existing_data.timestamp if existing_data else None
        
        # If we have existing data, only fetch from the last timestamp
        if last_timestamp:
            start_date = last_timestamp + timedelta(days=1)
            if start_date >= end_date:
                print("Price data is up to date")
                return
        
        # Fetch daily data from start_date until 7 days ago
        if start_date < minute_data_start:
            daily_df = yf_ticker.history(start=start_date, end=minute_data_start, interval="1d")
            if not daily_df.empty:
                for index, row in daily_df.iterrows():
                    # Check if we already have this timestamp
                    existing = db.query(StockPrice).filter(
                        StockPrice.stock_id == stock_id,
                        StockPrice.timestamp == index
                    ).first()
                    
                    if not existing:
                        price = StockPrice(
                            stock_id=stock_id,
                            timestamp=index,
                            open=row["Open"],
                            high=row["High"],
                            low=row["Low"],
                            close=row["Close"],
                            volume=row["Volume"]
                        )
                        db.add(price)
                db.commit()
        
        # Fetch minute data for the last 7 days
        minute_df = yf_ticker.history(start=minute_data_start, end=end_date, interval="1m")
        if not minute_df.empty:
            for index, row in minute_df.iterrows():
                # Check if we already have this timestamp
                existing = db.query(StockPrice).filter(
                    StockPrice.stock_id == stock_id,
                    StockPrice.timestamp == index
                ).first()
                
                if not existing:
                    price = StockPrice(
                        stock_id=stock_id,
                        timestamp=index,
                        open=row["Open"],
                        high=row["High"],
                        low=row["Low"],
                        close=row["Close"],
                        volume=row["Volume"]
                    )
                    db.add(price)
            db.commit()
                
    except Exception as e:
        print(f"Error fetching price data: {str(e)}")
        db.rollback()

@retry_on_exception(retries=3)
def fetch_market_data(yf_ticker, stock_id, db):
    """Fetch and store market context data."""
    info = yf_ticker.info

    # Get market cap in billions
    market_cap_raw = info.get('marketCap')
    market_cap = round(market_cap_raw / 1000000000, 2) if market_cap_raw is not None else None
    
    market_data = MarketData(
        stock_id=stock_id,
        date=datetime.now().date(),
        beta=info.get('beta'),
        sector=info.get('sector'),
        industry=info.get('industry'),
        float_shares=info.get('floatShares'),
        shares_outstanding=info.get('sharesOutstanding'),
        market_cap=market_cap,
    )
    db.add(market_data)
    db.commit()

@retry_on_exception(retries=3)
def fetch_options_data(yf_ticker, stock_id, db):
    """Fetch and store options data."""
    try:
        options = yf_ticker.options
        if options:
            chain = yf_ticker.option_chain(options[0])  # Get nearest expiration
            if hasattr(chain, 'puts') and hasattr(chain, 'calls'):
                # Calculate volumes
                put_volume = chain.puts['volume'].sum() if 'volume' in chain.puts.columns else 0
                call_volume = chain.calls['volume'].sum() if 'volume' in chain.calls.columns else 0
                total_volume = int(put_volume + call_volume)
                
                # Calculate implied volatility
                put_iv = chain.puts['impliedVolatility'].mean() if 'impliedVolatility' in chain.puts.columns else None
                call_iv = chain.calls['impliedVolatility'].mean() if 'impliedVolatility' in chain.calls.columns else None
                implied_volatility = (put_iv + call_iv) / 2 if pd.notnull(put_iv) and pd.notnull(call_iv) else None
                
                options_data = OptionsData(
                    stock_id=stock_id,
                    date=datetime.now().date(),
                    implied_volatility=implied_volatility,
                    put_call_ratio=put_volume/call_volume if call_volume > 0 else None,
                    options_volume=total_volume
                )
                db.add(options_data)
                db.commit()
    except Exception as e:
        print(f"Error fetching options data: {str(e)}")
        db.rollback()

def populate_stock_data(ticker: str, start_date: datetime = datetime(2021, 1, 1), end_date: datetime = datetime.now()):
    """Fetch and store stock data for the given ticker."""
    db = SessionLocal()
    
    try:
        # Check if stock already exists
        existing_stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        if existing_stock:
            print(f"Stock {ticker} already exists. Checking for missing data...")
            stock = existing_stock
        else:
            # Create stock entry
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            stock = Stock(
                ticker=ticker.upper(),
                name=info.get('longName', ticker.upper())
            )
            db.add(stock)
            db.commit()
            db.refresh(stock)

        yf_ticker = yf.Ticker(ticker)
        
        # Fetch and store price data
        fetch_price_data(yf_ticker, stock.id, db, start_date, end_date)
        
        # Store income statements (annual only)
        income_stmt_yearly = yf_ticker.get_income_stmt(freq='yearly', as_dict=True)
        if income_stmt_yearly:
            db.query(IncomeStatement).filter(IncomeStatement.stock_id == stock.id).delete()
            
            for date_str, data in income_stmt_yearly.items():
                try:
                    date = date_str.date() if isinstance(date_str, pd.Timestamp) else datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    if date.year < 2021 or date.year > 2024:
                        continue
                            
                    stmt = IncomeStatement(
                        stock_id=stock.id,
                        calendar_date=date,
                        period=ReportPeriod.ANNUAL,
                        total_revenue=data.get('TotalRevenue'),
                        net_income=data.get('NetIncome'),
                        operating_income=data.get('OperatingIncome'),
                        gross_profit=data.get('GrossProfit'),
                        operating_expense=data.get('OperatingExpense'),
                        ebit=data.get('EBIT'),
                        net_margin=data.get('NetIncome') / data.get('TotalRevenue') if data.get('NetIncome') and data.get('TotalRevenue') else None,
                        operating_margin=data.get('OperatingIncome') / data.get('TotalRevenue') if data.get('OperatingIncome') and data.get('TotalRevenue') else None
                    )
                    db.add(stmt)
                except Exception as e:
                    print(f"Error processing income statement for {date_str}: {e}")
            
            db.commit()
        
        # Store balance sheets (annual only)
        balance_sheet_yearly = yf_ticker.get_balance_sheet(freq='yearly', as_dict=True)
        if balance_sheet_yearly:
            db.query(BalanceSheet).filter(BalanceSheet.stock_id == stock.id).delete()
            
            for date_str, data in balance_sheet_yearly.items():
                try:
                    date = date_str.date() if isinstance(date_str, pd.Timestamp) else datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    if date.year < 2021 or date.year > 2024:
                        continue
                            
                    stmt = BalanceSheet(
                        stock_id=stock.id,
                        calendar_date=date,
                        period=ReportPeriod.ANNUAL,
                        total_assets=data.get('TotalAssets'),
                        total_liabilities=data.get('TotalLiabilitiesNetMinorityInterest'),
                        stockholders_equity=data.get('TotalEquityGrossMinorityInterest'),
                        cash_and_equivalents=data.get('CashAndCashEquivalents'),
                        total_debt=data.get('TotalDebt'),
                        current_assets=data.get('CurrentAssets'),
                        current_liabilities=data.get('CurrentLiabilities')
                    )
                    db.add(stmt)
                except Exception as e:
                    print(f"Error processing balance sheet for {date_str}: {e}")
            
            db.commit()
        
        # Store cash flow statements (annual only)
        cash_flow = yf_ticker.get_cashflow()
        if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
            db.query(CashFlowStatement).filter(CashFlowStatement.stock_id == stock.id).delete()
            
            for col in cash_flow.columns:
                if col.year >= 2021:
                    stmt_date = col.to_pydatetime().date()
                    
                    db.add(CashFlowStatement(
                        stock_id=stock.id,
                        calendar_date=stmt_date,
                        period=ReportPeriod.ANNUAL,
                        operating_cash_flow=cash_flow.loc["OperatingCashFlow", col] if "OperatingCashFlow" in cash_flow.index else None,
                        free_cash_flow=cash_flow.loc["FreeCashFlow", col] if "FreeCashFlow" in cash_flow.index else None
                    ))
            db.commit()
        
        # Store insider transactions
        transactions = yf_ticker.get_insider_transactions()
        if isinstance(transactions, pd.DataFrame) and not transactions.empty:
            db.query(InsiderTransaction).filter(InsiderTransaction.stock_id == stock.id).delete()
            
            for index, row in transactions.iterrows():
                date_val = pd.to_datetime(row.get('Start Date')).date() if pd.notnull(row.get('Start Date')) else datetime.now().date()
                
                text = str(row.get('Text', '')).strip()
                transaction_type = None
                transaction_price = None
                
                if 'sale at price' in text.lower():
                    transaction_type = 'Sale'
                    price_text = text.lower()[text.lower().find('at price') + 9:text.lower().find('per share')].strip()
                    try:
                        transaction_price = float(price_text)
                    except ValueError:
                        transaction_price = None
                elif 'purchase at price' in text.lower():
                    transaction_type = 'Purchase'
                    price_text = text.lower()[text.lower().find('at price') + 9:text.lower().find('per share')].strip()
                    try:
                        transaction_price = float(price_text)
                    except ValueError:
                        transaction_price = None
                
                if not transaction_type:
                    raw_type = str(row.get('Transaction', '')).strip().lower()
                    if raw_type in ['sale', 'sold']:
                        transaction_type = 'Sale'
                    elif raw_type in ['purchase', 'bought', 'buy']:
                        transaction_type = 'Purchase'
                
                if transaction_price is None:
                    shares = row.get('Shares')
                    value = row.get('Value')
                    if pd.notnull(shares) and pd.notnull(value) and shares != 0:
                        transaction_price = value / shares
                
                if transaction_type and (pd.notnull(row.get('Shares')) or pd.notnull(row.get('Value'))):
                    db.add(InsiderTransaction(
                        stock_id=stock.id,
                        date=date_val,
                        insider_name=row.get('Insider', ''),
                        transaction_type=transaction_type,
                        shares=row.get('Shares') if pd.notnull(row.get('Shares')) else None,
                        value=row.get('Value') if pd.notnull(row.get('Value')) else None,
                        transaction_price=transaction_price
                    ))
            db.commit()
        
        # Store analyst grades
        grades = yf_ticker.get_upgrades_downgrades()
        if isinstance(grades, pd.DataFrame) and not grades.empty:
            db.query(AnalystGrade).filter(AnalystGrade.stock_id == stock.id).delete()
            
            for index, row in grades.iterrows():
                action = None
                action_str = str(row.get("Action", "")).lower()
                
                if "upgrade" in action_str or action_str == "up":
                    action = GradeAction.UPGRADE
                elif "downgrade" in action_str or action_str == "down":
                    action = GradeAction.DOWNGRADE
                else:
                    action = GradeAction.MAINTAIN
                
                grade_date = index.date() if hasattr(index, 'date') else index
                
                db.add(AnalystGrade(
                    stock_id=stock.id,
                    grade_date=grade_date,
                    firm=row.get("Firm"),
                    to_grade=row.get("ToGrade"),
                    from_grade=row.get("FromGrade"),
                    action=action
                ))
            db.commit()
        
        # Store insider sentiment
        purchases = yf_ticker.get_insider_purchases()
        if isinstance(purchases, pd.DataFrame) and not purchases.empty:
            db.query(InsiderSentiment).filter(InsiderSentiment.stock_id == stock.id).delete()
            
            try:
                purchase_shares = float(purchases.loc[0, 'Shares']) if 'Shares' in purchases.columns and pd.notnull(purchases.loc[0, 'Shares']) else 0
                sale_shares = float(purchases.loc[1, 'Shares']) if len(purchases) > 1 and pd.notnull(purchases.loc[1, 'Shares']) else 0
                net_shares = purchase_shares - sale_shares
                total_volume = purchase_shares + sale_shares
                
                db.add(InsiderSentiment(
                    stock_id=stock.id,
                    date=datetime.now().date(),
                    net_shares=net_shares,
                    net_shares_pct=(net_shares / total_volume * 100) if total_volume > 0 else 0,
                    buy_shares_pct=(purchase_shares / total_volume * 100) if total_volume > 0 else 0,
                    sell_shares_pct=(sale_shares / total_volume * 100) if total_volume > 0 else 0,
                    total_transactions=int(purchases.loc[0, 'Trans']) + int(purchases.loc[1, 'Trans']) if 'Trans' in purchases.columns else 0
                ))
                db.commit()
            except Exception as e:
                print(f"Error calculating insider sentiment: {str(e)}")
        
        # Store financial ratios
        info = yf_ticker.info
        db.query(FinancialRatios).filter(FinancialRatios.stock_id == stock.id).delete()
        
        # Calculate free cash flow per share
        free_cash_flow = info.get('freeCashflow')
        shares_outstanding = info.get('sharesOutstanding')
        free_cash_flow_per_share = free_cash_flow / shares_outstanding if free_cash_flow and shares_outstanding else None
        
        
        ratios = FinancialRatios(
            stock_id=stock.id,
            date=datetime.now().date(),
            pe_ratio=info.get("trailingPE"),
            pb_ratio=info.get("priceToBook"),
            ps_ratio=info.get("priceToSalesTrailing12Months"),
            debt_to_equity=info.get("debtToEquity"),
            current_ratio=info.get("currentRatio"),
            return_on_equity=info.get("returnOnEquity"),
            profit_margins=info.get("profitMargins"),
            operating_margins=info.get("operatingMargins"),
            beta=info.get("beta"),
            earnings_growth=info.get("earningsGrowth"),
            revenue_growth=info.get("revenueGrowth"),
            free_cash_flow=free_cash_flow,
            free_cash_flow_per_share=free_cash_flow_per_share,
            earnings_per_share=info.get('trailingEps')
        )
        db.add(ratios)
        db.commit()
        
        # Fetch additional data
        fetch_market_data(yf_ticker, stock.id, db)
        fetch_options_data(yf_ticker, stock.id, db)

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        db.rollback()
    finally:
        db.close()

def main():
    # Drop all tables
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    
    # Populate data for some common stocks
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()
    
    for ticker in tickers:
        print(f"Populating data for {ticker}...")
        populate_stock_data(ticker, start_date=start_date, end_date=end_date)
        print(f"Completed {ticker}")

if __name__ == "__main__":
    main()
