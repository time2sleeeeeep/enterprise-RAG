"""mocked 集成测试：check_health 全 up / 部分 down / 超时（H6）。"""

import asyncio
import pytest


def test_check_health_all_up(monkeypatch):
    """三个探活全 up → all_up=True。"""
    import src.api.health as h

    monkeypatch.setattr(h, "_ping_milvus", lambda: "up")
    monkeypatch.setattr(h, "_ping_mysql", lambda: "up")

    async def redis_up():
        return "up"

    monkeypatch.setattr(h, "_ping_redis", redis_up)

    r = asyncio.run(h.check_health())
    assert r["all_up"] is True
    assert r["dependencies"] == {"milvus": "up", "mysql": "up", "redis": "up"}


def test_check_health_milvus_down(monkeypatch):
    """Milvus down → all_up=False, 其他仍 up。"""
    import src.api.health as h

    monkeypatch.setattr(h, "_ping_milvus", lambda: "down: MilvusException")
    monkeypatch.setattr(h, "_ping_mysql", lambda: "up")

    async def redis_up():
        return "up"

    monkeypatch.setattr(h, "_ping_redis", redis_up)

    r = asyncio.run(h.check_health())
    assert r["all_up"] is False
    assert r["dependencies"]["milvus"].startswith("down")
    assert r["dependencies"]["mysql"] == "up"
    assert r["dependencies"]["redis"] == "up"


def test_check_health_redis_timeout(monkeypatch):
    """Redis 超时 (check_health 内部 wait_for 2s 超时生效)。"""
    import src.api.health as h

    monkeypatch.setattr(h, "_ping_milvus", lambda: "up")
    monkeypatch.setattr(h, "_ping_mysql", lambda: "up")
    # 注：不 patch _ping_redis，让它内部 wait_for 正常运作；
    # 只 patch get_redis 返回一个会 sleep 5s 的假客户端
    async def fake_get_redis():
        class SlowClient:
            async def ping(self):
                await asyncio.sleep(5)
                return True

        return SlowClient()

    monkeypatch.setattr(h, "get_redis", fake_get_redis)

    r = asyncio.run(h.check_health())
    assert r["all_up"] is False
    assert "timeout" in r["dependencies"]["redis"]
