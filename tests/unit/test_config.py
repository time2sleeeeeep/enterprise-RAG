"""测试 config.py cors_origins 解析（逗号分隔 / JSON 数组 / 默认）。"""

import pytest
from src.config import Settings


class TestCorsOrigins:
    def test_default_wildcard(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        s = Settings()
        assert s.cors_origins == ["*"]

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com")
        s = Settings()
        assert s.cors_origins == ["http://a.com", "http://b.com"]

    def test_json_array(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", '["http://x.com","http://y.com"]')
        s = Settings()
        assert s.cors_origins == ["http://x.com", "http://y.com"]

    def test_single_value(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://solo.com")
        s = Settings()
        assert s.cors_origins == ["http://solo.com"]
