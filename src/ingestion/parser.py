import re


def clean_text(text: str) -> str:
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
