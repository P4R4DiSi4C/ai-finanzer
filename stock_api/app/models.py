from enum import Enum
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from .database import Base

class ReportPeriod(str, Enum):
    ANNUAL = "ANNUAL"    # For yearly reports only, no need for quarterly

class GradeAction(str, Enum):
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    MAINTAIN = "maintain"

class Stock(Base):
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True)
    name = Column(String)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    income_statements = relationship("IncomeStatement", back_populates="stock")
    cash_flow_statements = relationship("CashFlowStatement", back_populates="stock")
    insider_transactions = relationship("InsiderTransaction", back_populates="stock")
    analyst_grades = relationship("AnalystGrade", back_populates="stock")
    insider_sentiments = relationship("InsiderSentiment", back_populates="stock")
    market_data = relationship("MarketData", back_populates="stock")
    options_data = relationship("OptionsData", back_populates="stock")
    balance_sheets = relationship("BalanceSheet", back_populates="stock")
    financial_ratios = relationship("FinancialRatios", back_populates="stock")

class StockPrice(Base):
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    stock = relationship("Stock", back_populates="prices")

class IncomeStatement(Base):
    __tablename__ = "income_statements"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    calendar_date = Column(Date)
    period = Column(SQLEnum(ReportPeriod))
    total_revenue = Column(Float)
    net_income = Column(Float)
    operating_income = Column(Float)
    gross_profit = Column(Float)
    operating_expense = Column(Float)
    ebit = Column(Float)
    net_margin = Column(Float)
    operating_margin = Column(Float)
    
    stock = relationship("Stock", back_populates="income_statements")

class BalanceSheet(Base):
    __tablename__ = "balance_sheets"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    calendar_date = Column(Date)
    period = Column(SQLEnum(ReportPeriod))
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    stockholders_equity = Column(Float)
    cash_and_equivalents = Column(Float)
    total_debt = Column(Float)
    current_assets = Column(Float)
    current_liabilities = Column(Float)
    
    stock = relationship("Stock", back_populates="balance_sheets")

class CashFlowStatement(Base):
    __tablename__ = "cash_flow_statements"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    calendar_date = Column(Date)
    period = Column(SQLEnum(ReportPeriod))
    operating_cash_flow = Column(Float)
    free_cash_flow = Column(Float)
    
    stock = relationship("Stock", back_populates="cash_flow_statements")

class InsiderTransaction(Base):
    __tablename__ = "insider_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date)
    insider_name = Column(String)
    transaction_type = Column(String)
    shares = Column(Float)
    value = Column(Float)
    transaction_price = Column(Float)
    
    stock = relationship("Stock", back_populates="insider_transactions")

class AnalystGrade(Base):
    __tablename__ = "analyst_grades"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    grade_date = Column(Date)
    firm = Column(String)
    to_grade = Column(String)
    from_grade = Column(String)
    action = Column(SQLEnum(GradeAction))
    
    stock = relationship("Stock", back_populates="analyst_grades")

class InsiderSentiment(Base):
    __tablename__ = "insider_sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date)
    net_shares = Column(Float)
    net_shares_pct = Column(Float)
    buy_shares_pct = Column(Float)
    sell_shares_pct = Column(Float)
    total_transactions = Column(Integer)
    
    stock = relationship("Stock", back_populates="insider_sentiments")

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date)
    beta = Column(Float)
    sector = Column(String)
    industry = Column(String)
    float_shares = Column(Float)
    shares_outstanding = Column(Float)
    market_cap = Column(Float)  # Market cap in billions of dollars
    
    stock = relationship("Stock", back_populates="market_data")

class FinancialRatios(Base):
    __tablename__ = "financial_ratios"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    return_on_equity = Column(Float)
    profit_margins = Column(Float)
    operating_margins = Column(Float)
    beta = Column(Float)
    earnings_growth = Column(Float)
    revenue_growth = Column(Float)
    free_cash_flow = Column(Float)
    free_cash_flow_per_share = Column(Float)
    earnings_per_share = Column(Float)
    
    stock = relationship("Stock", back_populates="financial_ratios")

class OptionsData(Base):
    __tablename__ = "options_data"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date)
    implied_volatility = Column(Float)
    put_call_ratio = Column(Float)
    options_volume = Column(Integer)
    
    stock = relationship("Stock", back_populates="options_data")
