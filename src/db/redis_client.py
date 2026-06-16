import redis.asyncio as redis

from src.config import settings

redis_pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=True)


async def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=redis_pool)


async def close_redis():
    await redis_pool.aclose()
