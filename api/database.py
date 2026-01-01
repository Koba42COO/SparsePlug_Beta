"""
Database Connection Management
==============================

Provides async database connection pooling and session management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from api.config import settings


# Create async engine with connection pooling
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=settings.debug,  # Log SQL in debug mode
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes.
    
    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


async def init_database():
    """Initialize database connection pool."""
    # Test connection
    async with engine.begin() as conn:
        await conn.execute("SELECT 1")
    print("✅ Database connection established")


async def close_database():
    """Close database connection pool."""
    await engine.dispose()
    print("✅ Database connection pool closed")


# Health check
async def check_database_health() -> dict:
    """Check database connectivity."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            _ = result.scalar()
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class DatabaseManager:
    """
    Database manager for advanced operations.
    
    Provides:
    - Connection health monitoring
    - Pool statistics
    - Transaction management
    """
    
    def __init__(self):
        self.engine = engine
    
    async def get_pool_status(self) -> dict:
        """Get connection pool status."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0,
        }
    
    async def execute_raw(self, sql: str) -> list:
        """Execute raw SQL (use with caution)."""
        async with engine.begin() as conn:
            result = await conn.execute(sql)
            return result.fetchall()
    
    @asynccontextmanager
    async def transaction(self):
        """
        Explicit transaction context.
        
        Usage:
            async with db_manager.transaction() as session:
                # Multiple operations in single transaction
                await session.execute(...)
                await session.execute(...)
                # Auto-commits on success, rolls back on exception
        """
        async with async_session_factory() as session:
            async with session.begin():
                yield session


# Singleton instance
db_manager = DatabaseManager()
