# 文本解析模块：提供文本清洗（去除多余空行/空格）和 [PAGE N] 页面标记提取两个工具函数。

import re


def clean_text(text: str) -> str:
    """清洗文本：合并连续空行、压缩水平空白、去除行首行尾空格。"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\n )+", "\n", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        else:
            lines.append("")
    return "\n".join(lines).strip()


def extract_page_markers(text: str) -> list[tuple[int, str]]:
    """Split text by [PAGE N] markers, returning (page_num, content) pairs."""
    pattern = r"\[PAGE (\d+)\]\n"
    parts = re.split(pattern, text)
    if len(parts) == 1:
        return [(0, text)]

    results = []
    i = 1
    while i < len(parts):
        page_num = int(parts[i])
        content = parts[i + 1] if i + 1 < len(parts) else ""
        results.append((page_num, content.strip()))
        i += 2
    return results
