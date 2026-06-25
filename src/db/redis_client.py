# Redis 客户端：创建异步连接池，提供获取连接和关闭连接池的工具函数。

import redis.asyncio as redis

from src.config import settings

redis_pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=True)


async def get_redis() -> redis.Redis:
    """从连接池获取一个 Redis 异步客户端实例。"""
    return redis.Redis(connection_pool=redis_pool)


async def close_redis():
    """关闭 Redis 连接池，在应用关闭时调用。"""
    await redis_pool.aclose()
