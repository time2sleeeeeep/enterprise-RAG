"""测试 ingestion/parser.py 纯函数。"""

from src.ingestion.parser import clean_text, extract_page_markers


class TestCleanText:
    def test_collapse_multiple_newlines(self):
        assert clean_text("a\n\n\nb") == "a\n\nb"

    def test_compress_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_strip_trailing_whitespace(self):
        assert clean_text("  hi  \n  ") == "hi"

    def test_preserve_single_newline(self):
        assert clean_text("line1\nline2") == "line1\nline2"


class TestExtractPageMarkers:
    def test_no_markers(self):
        assert extract_page_markers("plain text") == [(0, "plain text")]

    def test_single_page(self):
        assert extract_page_markers("[PAGE 1]\ncontent") == [(1, "content")]

    def test_multiple_pages(self):
        text = "[PAGE 1]\nfirst\n[PAGE 3]\nthird"
        result = extract_page_markers(text)
        assert len(result) == 2
        assert result[0] == (1, "first")
        assert result[1] == (3, "third")
