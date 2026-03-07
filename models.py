from sqlalchemy import Table, Column, Integer, String, Float, MetaData, text
from sqlalchemy.orm import declarative_base

Base = declarative_base()
metadata = Base.metadata

class KV(Base):
    __tablename__ = 'kv'
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)

class ClosedTrade(Base):
    __tablename__ = 'closed_trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Integer)
    market_slug = Column(String)
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl_usd = Column(Float)
    outcome_win = Column(Integer)
    # Adding Phase 3 fields
    slippage = Column(Float)
    exit_reason = Column(String)
    # Adding Phase 4 fields
    regime = Column(String)
    features = Column(String)  # JSON string
    kelly_fraction = Column(Float)
