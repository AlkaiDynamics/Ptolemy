"""
Database module for PTOLEMY using SQLAlchemy with async support.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Float, Integer
import datetime
from pathlib import Path

from ptolemy.config import BASE_DIR

# Create async engine
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/data/ptolemy.db"
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Define models
class Event(Base):
    __tablename__ = "events"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    type = Column(String, index=True)
    data = Column(JSON)
    
class Relationship(Base):
    __tablename__ = "relationships"
    
    id = Column(String, primary_key=True)
    source_entity = Column(String, index=True)
    target_entity = Column(String, index=True)
    relationship_type = Column(String, index=True)
    meta_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.now)

class Pattern(Base):
    __tablename__ = "patterns"
    
    id = Column(String, primary_key=True)
    name = Column(String, index=True)
    type = Column(String, index=True)
    implementation = Column(String)
    meta_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.now)

class Insight(Base):
    __tablename__ = "insights"
    
    id = Column(String, primary_key=True)
    type = Column(String, index=True)
    content = Column(String)
    relevance = Column(Float)
    meta_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.now)

async def init_db():
    """Initialize the database by creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Dependency for FastAPI to get a database session."""
    async with async_session() as session:
        yield session
