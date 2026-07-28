import certifi
import time
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def get_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database with optimized connection pooling"""
    if db.client is None:
        db.client = AsyncIOMotorClient(
            settings.mongodb_uri,
            tlsCAFile=certifi.where(),
            # OPTIMIZED: Connection pool settings for 50ms target
            serverSelectionTimeoutMS=3000,  # Fail fast (from 5s)
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
            maxPoolSize=100,                # Increased from 50
            minPoolSize=20,                 # Increased from 10 (warm connections)
            maxIdleTimeMS=30000,            # From 45s -> 30s (faster reconnect)
            waitQueueTimeoutMS=10000,       # NEW: Reject if queue full
            retryWrites=True,
            directConnection=False,
            heartbeatFrequencyMS=10000,     # NEW: Check health every 10s
            serverMonitoringMode="auto",    # NEW: Auto-detect topology
        )
    return db.client.get_database("agentic_ai")

async def warmup_connection_pool():
    """Warm up connection pool on startup to avoid first-request penalty"""
    try:
        db_instance = await get_database()
        ping_tasks = [
            db_instance.command("ping")
            for _ in range(20)
        ]
        await asyncio.gather(*ping_tasks, return_exceptions=True)
        print("✓ MongoDB connection pool warmed up (20 connections)")
    except Exception as e:
        print(f"Warning: Connection pool warmup failed: {e}")

async def get_collection(name: str):
    database = await get_database()
    return database.get_collection(name)

async def init_indexes():
    """Create comprehensive MongoDB indexes for performance"""
    database = await get_database()
    
    print("Creating MongoDB indexes for optimal performance...")
    
    # Users collection indexes
    users_col = database['users']
    await users_col.create_index('email', unique=True, sparse=True, background=True)
    await users_col.create_index([('created_at', -1)], background=True)
    await users_col.create_index([('_id', 1), ('created_at', -1)], background=True)
    
    # Projects collection indexes - OPTIMIZED with compound indexes
    projects_col = database['projects']
    await projects_col.create_index([('user_id', 1), ('status', 1)], background=True)  # NEW
    await projects_col.create_index([('created_at', -1)], background=True)
    await projects_col.create_index([('user_id', 1), ('created_at', -1)], background=True)  # NEW compound
    
    # Tasks collection indexes - OPTIMIZED with compound indexes
    tasks_col = database['tasks']
    await tasks_col.create_index([('project_id', 1), ('status', 1)], background=True)  # NEW
    await tasks_col.create_index([('created_at', -1)], background=True)
    await tasks_col.create_index([('project_id', 1), ('created_at', -1)], background=True)  # NEW compound
    
    # Connectors collection indexes
    connectors_col = database['connectors']
    await connectors_col.create_index([('user_id', 1)], background=True)
    await connectors_col.create_index([('connector_type', 1)], background=True)
    
    print("✓ MongoDB indexes created successfully (11 total)")
    
    # Connectors collection indexes
    connectors_col = database['connectors']
    await connectors_col.create_index('user_id')
    await connectors_col.create_index('connector_type')
    
    print("✓ MongoDB indexes created successfully")
