"""测试 format_context（纯字符串格式）。"""

from src.core.generator import format_context


class TestFormatContext:
    def test_basic_numbering(self):
        docs = [
            {"content": "hello", "source": "a.md", "page_num": 0},
            {"content": "world", "source": "b.md", "page_num": 5},
        ]
        result = format_context(docs)
        assert "【参考1】[a.md]" in result
        assert "【参考2】[b.md, 第5页]" in result
        assert "hello" in result
        assert "world" in result

    def test_empty(self):
        assert format_context([]) == ""
