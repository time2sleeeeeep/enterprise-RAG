# /health 依赖探活模块：分别 ping Milvus / MySQL / Redis，任一不可用则视为 degraded。
# sync 探活（Milvus/MySQL）丢入独立线程池，避免挂死依赖耗尽主事件循环的默认执行器
# （chat.py 的 asyncio.to_thread 也用默认池，二者隔离，互不影响）。

import asyncio
from concurrent.futures import ThreadPoolExecutor

from pymilvus import utility
from sqlalchemy import text

from src.db.mysql_client import engine
from src.db.redis_client import get_redis

# 独立小线程池：与默认执行器隔离，挂死的探活不占用业务请求的 to_thread worker。
_probe_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="health-probe")
_PROBE_TIMEOUT = 2.0  # 单依赖探活超时（秒）


def _ping_milvus() -> str:
    """通过 list_collections 探活 Milvus；成功 'up'，异常 'down: <类型>'。"""
    try:
        utility.list_collections()
        return "up"
    except Exception as e:
        return f"down: {type(e).__name__}"


def _ping_mysql() -> str:
    """通过 SELECT 1 探活 MySQL。"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "up"
    except Exception as e:
        return f"down: {type(e).__name__}"


async def _ping_redis() -> str:
    """异步 ping Redis，带超时。"""
    try:
        client = await get_redis()
        await asyncio.wait_for(client.ping(), timeout=_PROBE_TIMEOUT)
        return "up"
    except asyncio.TimeoutError:
        return "down: timeout"
    except Exception as e:
        return f"down: {type(e).__name__}"


async def _probe_sync(fn) -> str:
    """把 sync 探活丢入独立线程池并加超时；超时返回 down（底层线程仍在后台跑但不影响主池）。"""
    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_probe_executor, fn),
            timeout=_PROBE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return "down: timeout"
    except Exception as e:
        return f"down: {type(e).__name__}"


async def check_health() -> dict:
    """并发探活三个依赖，返回 {dependencies, all_up}。"""
    milvus, mysql, redis_status = await asyncio.gather(
        _probe_sync(_ping_milvus),
        _probe_sync(_ping_mysql),
        _ping_redis(),
    )
    deps = {"milvus": milvus, "mysql": mysql, "redis": redis_status}
    return {"dependencies": deps, "all_up": all(v == "up" for v in deps.values())}
