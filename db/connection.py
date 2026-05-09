"""
Database connection — SQLAlchemy engine + session factory.

Used by: api/main.py, monitoring/score_monitor.py, spark/batch_inference.py

Provides both sync and async engines so FastAPI (async) and the Spark
post-write step (sync) can share the same connection config.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

# Settings loaded once at import
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous engine (used by Spark job, monitoring, data loading)
# ---------------------------------------------------------------------------
_sync_engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,  # verify connections before use
    echo=False,
)

SyncSessionLocal = sessionmaker(
    bind=_sync_engine,
    autocommit=False,
    autoflush=False,
)


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Context manager for a sync SQLAlchemy session."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Asynchronous engine (used by FastAPI)
# ---------------------------------------------------------------------------
_async_engine = create_async_engine(
    settings.database.async_url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = sessionmaker(
    bind=_async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for FastAPI route handlers."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# FastAPI dependency injection version
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ping_database() -> bool:
    """Check database connectivity. Returns True if reachable."""
    try:
        with _sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database ping failed: {e}")
        return False


def run_schema(schema_path: str = None) -> None:
    """
    Execute schema.sql against the database.
    Idempotent — uses CREATE IF NOT EXISTS throughout.
    """
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_path, "r") as f:
        sql = f.read()

    with _sync_engine.connect() as conn:
        # Split on semicolons but skip empty statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
    logger.info("Schema applied successfully.")
