# 评估报告生成模块：从评估结果 JSON 生成 HTML 和 Markdown 格式的可视化报告。
# HTML 报告为单文件自包含（内联 CSS + SVG 图表），无需外部依赖。

import json
import os
import statistics as stats_lib
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# 颜色主题
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#f8f9fa",
    "card_bg": "#ffffff",
    "primary": "#2563eb",
    "success": "#16a34a",
    "warning": "#ea580c",
    "danger": "#dc2626",
    "text": "#1e293b",
    "text_secondary": "#64748b",
    "border": "#e2e8f0",
    "chart_colors": ["#2563eb", "#16a34a", "#ea580c", "#8b5cf6", "#ec4899", "#06b6d4"],
}

SCORE_COLORS = {
    (0.80, 1.01): "#16a34a",   # 优秀：绿色
    (0.60, 0.80): "#ea580c",   # 一般：橙色
    (0.00, 0.60): "#dc2626",   # 差：红色
}


def _score_color(score: float) -> str:
    """根据分值返回颜色。"""
    for (low, high), color in SCORE_COLORS.items():
        if low <= score < high:
            return color
    return COLORS["text"]


def _format_score(score: float | None) -> str:
    """格式化分数显示。"""
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def _load_results_from_dir(results_dir: str) -> dict[str, dict]:
    """从目录加载所有 eval JSON 结果文件（ablation_summary.json 或 *_detail.json）。"""
    results: dict[str, dict] = {}
    dir_path = Path(results_dir)
    if not dir_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return results

    # 优先加载 summary
    for fpath in sorted(dir_path.glob("*.json")):
        if "summary" in fpath.name:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            results[k] = v
            break

    # 如果没有 summary，逐个加载 detail
    if not results:
        for fpath in sorted(dir_path.glob("*_detail.json")):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                cfg_name = data.get("config_name", fpath.stem.replace("_detail", ""))
                results[cfg_name] = data

    return results


def _sub_colors(template: str) -> str:
    """替换模板中的 {{color_name}} 占位符为实际颜色值（安全替换，不干扰 CSS 花括号）。"""
    result = template
    for key, value in COLORS.items():
        if isinstance(value, str):
            result = result.replace("{{" + key + "}}", value)
    # 将剩余的 {{ }} 恢复为单花括号（非颜色占位符的 CSS 大括号）
    result = result.replace("{{", "{").replace("}}", "}")
    return result


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------

_CSS_TEMPLATE = """<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: {{bg}}; color: {{text}}; padding: 24px; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 28px; margin-bottom: 4px; }}
h2 {{ font-size: 20px; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 2px solid {{border}}; }}
h3 {{ font-size: 16px; margin: 16px 0 8px; }}
.subtitle {{ color: {{text_secondary}}; font-size: 14px; margin-bottom: 24px; }}
.card {{ background: {{card_bg}}; border-radius: 8px; padding: 20px; margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid {{border}}; }}
th {{ font-weight: 600; background: {{bg}}; color: {{text_secondary}}; font-size: 13px; text-transform: uppercase; }}
tr:hover td {{ background: {{bg}}; }}
.score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
               color: white; font-weight: 600; font-size: 12px; min-width: 56px; text-align: center; }}
.best {{ font-weight: 700; }}
.footer {{ text-align: center; color: {{text_secondary}}; font-size: 12px; margin-top: 48px; padding-top: 16px; }}
.metric-section {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.metric-card {{ background: {{card_bg}}; border-radius: 8px; padding: 16px; text-align: center;
               box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.metric-card .value {{ font-size: 32px; font-weight: 700; }}
.metric-card .label {{ font-size: 12px; color: {{text_secondary}}; text-transform: uppercase; margin-top: 4px; }}
svg {{ max-width: 100%; }}
</style>"""

_CSS = _sub_colors(_CSS_TEMPLATE)


def _html_header(title: str, subtitle: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{_CSS}
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
"""


def _html_footer() -> str:
    return f"""<div class="footer">
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Enterprise RAG Evaluation Framework
</div></div></body></html>"""


def _render_metric_table(results: dict[str, dict]) -> str:
    """渲染指标对比表。"""
    if not results:
        return "<p>No results available.</p>"

    # 收集所有指标名
    all_metrics = set()
    for cfg in results.values():
        metrics = cfg.get("metrics", cfg.get("metric_stats", {}))
        if isinstance(metrics, dict):
            all_metrics.update(metrics.keys())
        elif isinstance(metrics, list):
            pass

    # 区分 LLM 指标和检索指标
    llm_metrics = sorted([m for m in all_metrics if not m.startswith(("precision_", "recall_", "mrr", "ndcg_", "hit_rate", "map_"))])
    retrieval_metrics = sorted([m for m in all_metrics if m in ("precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "hit_rate_at_5", "map_score")])
    ordered = llm_metrics + retrieval_metrics

    if not ordered:
        return "<p>No metrics found in results.</p>"

    rows = []
    # 表头
    header_cells = "<th>Config</th>" + "".join(f"<th>{m}</th>" for m in ordered)
    rows.append(f"<tr>{header_cells}</tr>")

    for cfg_name, cfg_data in results.items():
        metrics = cfg_data.get("metrics", {})
        cells = f"<td><strong>{cfg_name}</strong></td>"
        for m in ordered:
            val = metrics.get(m)
            if val is None:
                cells += "<td>—</td>"
            else:
                color = _score_color(val)
                cells += f'<td><span class="score-badge" style="background:{color}">{val:.4f}</span></td>'
        rows.append(f"<tr>{cells}</tr>")

    # 找最佳值
    best_row = "<td><strong>Best</strong></td>"
    for m in ordered:
        valid_vals = [(n, d["metrics"].get(m)) for n, d in results.items() if d.get("metrics", {}).get(m) is not None]
        if valid_vals:
            best_name, best_val = max(valid_vals, key=lambda x: x[1])
            best_row += f'<td><span class="best">{best_val:.4f}</span> <small>({best_name})</small></td>'
        else:
            best_row += "<td>—</td>"
    rows.append(f"<tr style=\"background:{COLORS['bg']}; font-size:13px\">{best_row}</tr>")

    return f"""
<div class="card">
<h3>📊 Metric Comparison</h3>
<table>{''.join(rows)}</table>
</div>"""


def _render_latency_table(results: dict[str, dict]) -> str:
    """渲染延迟分解表。"""
    timing_keys = [
        ("embedding_ms", "Embedding"),
        ("dense_search_ms", "Dense Search"),
        ("sparse_search_ms", "Sparse Search"),
        ("reranking_ms", "Reranking"),
        ("generation_ms", "Generation"),
        ("scoring_ms", "Scoring"),
    ]

    has_timing = False
    for cfg_data in results.values():
        if cfg_data.get("per_component_timing") or cfg_data.get("latency_stats"):
            has_timing = True
            break

    if not has_timing:
        return ""

    rows = []
    header = "<th>Config</th>"
    for _, label in timing_keys:
        header += f"<th>{label} (ms)</th>"
    header += "<th>Total (ms)</th><th>Avg/Sample (s)</th>"
    rows.append(f"<tr>{header}</tr>")

    for cfg_name, cfg_data in results.items():
        timing = cfg_data.get("per_component_timing", {})
        cells = f"<td><strong>{cfg_name}</strong></td>"
        total = 0.0
        for key, _ in timing_keys:
            val = timing.get(key, 0) if isinstance(timing, dict) else 0
            cells += f"<td>{val:.0f}</td>"
            total += val
        cells += f"<td><strong>{total:.0f}</strong></td>"
        avg_s = cfg_data.get("avg_latency", 0)
        cells += f"<td>{avg_s:.2f}</td>"
        rows.append(f"<tr>{cells}</tr>")

    return f"""
<div class="card">
<h3>⏱️ Latency Breakdown</h3>
<table>{''.join(rows)}</table>
</div>"""


def _render_svg_bar_chart(
    labels: list[str],
    datasets: list[dict[str, list[float] | str]],
    title: str = "",
    width: int = 800,
    height: int = 400,
) -> str:
    """生成内联 SVG 柱状图。

    Args:
        labels: X 轴标签
        datasets: [{label: str, values: [float], color: str}, ...]
        title: 图表标题
    """
    if not labels or not datasets:
        return ""

    margin = {"top": 40, "right": 20, "bottom": 80, "left": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    # 找最大值
    max_val = 0.0
    for ds in datasets:
        vals = ds["values"]
        if isinstance(vals, list):
            max_val = max(max_val, max(vals) if vals else 0)
    if max_val == 0:
        max_val = 1.0
    max_val = max_val * 1.1  # 留 10% 空间

    num_groups = len(labels)
    num_bars = len(datasets)
    group_width = plot_w / max(num_groups, 1)
    bar_width = group_width / (num_bars + 1) * 0.7
    bar_gap = group_width / (num_bars + 1) * 0.3

    bars = []
    for gi, label in enumerate(labels):
        for bi, ds in enumerate(datasets):
            vals = ds["values"]
            val = vals[gi] if isinstance(vals, list) and gi < len(vals) else 0
            bar_h = (val / max_val) * plot_h if max_val > 0 else 0
            x = margin["left"] + gi * group_width + bar_gap + bi * (bar_width + bar_gap)
            y = margin["top"] + plot_h - bar_h
            color = ds.get("color", COLORS["chart_colors"][bi % len(COLORS["chart_colors"])])
            label_text = f"{val:.3f}"
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}" rx="2" />'
                f'<text x="{x + bar_width/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{COLORS["text"]}">{label_text}</text>'
            )

    # Y 轴刻度
    y_ticks = []
    for i in range(6):
        tick_val = max_val * i / 5
        y = margin["top"] + plot_h - (tick_val / max_val) * plot_h
        y_ticks.append(
            f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w:.1f}" '
            f'y2="{y:.1f}" stroke="{COLORS["border"]}" stroke-dasharray="4,4" />'
            f'<text x="{margin["left"] - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{COLORS["text_secondary"]}">{tick_val:.2f}</text>'
        )

    # X 轴标签
    x_labels = []
    for gi, label in enumerate(labels):
        x = margin["left"] + gi * group_width + group_width / 2
        x_labels.append(
            f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 20:.1f}" text-anchor="end" '
            f'font-size="11" fill="{COLORS["text"]}" transform="rotate(-30, {x:.1f}, {margin["top"] + plot_h + 20:.1f})">{label}</text>'
        )

    # 图例
    legend_y = margin["top"] + plot_h + 55
    legend_items = []
    for bi, ds in enumerate(datasets):
        x = margin["left"] + bi * 150
        color = ds.get("color", COLORS["chart_colors"][bi % len(COLORS["chart_colors"])])
        legend_items.append(
            f'<rect x="{x:.1f}" y="{legend_y:.1f}" width="12" height="12" fill="{color}" rx="2" />'
            f'<text x="{x + 18:.1f}" y="{legend_y + 11:.1f}" font-size="12" fill="{COLORS["text"]}">{ds["label"]}</text>'
        )

    return f"""
<div class="card">
<h3>{title}</h3>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
  {"".join(y_ticks)}
  <line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="{COLORS["border"]}" />
  <line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="{COLORS["border"]}" />
  {"".join(bars)}
  {"".join(x_labels)}
  {"".join(legend_items)}
</svg>
</div>"""


def generate_html_report(
    results: dict[str, dict],
    output_path: str,
    title: str = "RAG Evaluation Report",
) -> str:
    """生成自包含的 HTML 评估报告。

    Args:
        results: {config_name: {metrics, avg_latency, per_component_timing, ...}}
        output_path: 输出 HTML 文件路径
        title: 报告标题

    Returns:
        输出文件路径
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = f"Generated: {now} · Configs: {len(results)}"

    sections = [
        _html_header(title, subtitle),
        "<h2>📈 Overview</h2>",
        _render_summary_cards(results),
        "<h2>📊 Metrics</h2>",
        _render_metric_table(results),
    ]

    # 延迟分解
    latency_section = _render_latency_table(results)
    if latency_section:
        sections.append("<h2>⏱️ Latency</h2>")
        sections.append(latency_section)

    # 柱状图
    sections.append("<h2>📉 Charts</h2>")
    chart_html = _render_metric_charts(results)
    if chart_html:
        sections.append(chart_html)

    sections.append(_html_footer())

    html = "\n".join(sections)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_path}")
    return output_path


def _render_summary_cards(results: dict[str, dict]) -> str:
    """渲染概览卡片（每个配置的核心指标）。"""
    cards = []
    for cfg_name, cfg_data in results.items():
        metrics = cfg_data.get("metrics", {})
        key_metrics = []
        for m in ["faithfulness", "answer_relevancy", "correctness", "context_precision"]:
            val = metrics.get(m)
            if val is not None:
                key_metrics.append((m, val))
        metric_html = "".join(
            f'<div class="metric-card"><div class="value" style="color:{_score_color(v)}">{v:.3f}</div>'
            f'<div class="label">{m}</div></div>'
            for m, v in key_metrics
        )
        cards.append(f'<div class="card"><h3>{cfg_name}</h3><div class="metric-section">{metric_html}</div></div>')

    return "\n".join(cards)


def _render_metric_charts(results: dict[str, dict]) -> str:
    """生成指标对比柱状图。"""
    if not results:
        return ""

    # 收集所有指标
    all_metrics = set()
    for cfg in results.values():
        all_metrics.update(cfg.get("metrics", {}).keys())
    if not all_metrics:
        return ""

    # 选最重要的指标
    priority = [
        "faithfulness", "faithfulness_claim", "answer_relevancy",
        "context_precision", "context_recall", "correctness",
        "precision_at_5", "recall_at_10", "mrr", "ndcg_at_10",
    ]
    selected = [m for m in priority if m in all_metrics]
    # 补充其他指标
    for m in sorted(all_metrics):
        if m not in selected:
            selected.append(m)
    selected = selected[:10]  # 最多 10 个指标

    labels = selected
    config_names = list(results.keys())
    datasets = []
    for ci, cfg_name in enumerate(config_names):
        metrics = results[cfg_name].get("metrics", {})
        values = [metrics.get(m, 0) or 0 for m in selected]
        datasets.append({
            "label": cfg_name,
            "values": values,
            "color": COLORS["chart_colors"][ci % len(COLORS["chart_colors"])],
        })

    return _render_svg_bar_chart(labels, datasets, title="📊 Metric Comparison Chart")


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def generate_markdown_report(
    results: dict[str, dict],
    output_path: str,
    title: str = "RAG Evaluation Report",
) -> str:
    """生成 Markdown 格式的评估报告。

    Args:
        results: {config_name: {metrics, avg_latency, per_component_timing, ...}}
        output_path: 输出 .md 文件路径
        title: 报告标题

    Returns:
        输出文件路径
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"",
        f"> Generated: {now} · Configs: {len(results)}",
        f"",
    ]

    # 指标表
    all_metrics = set()
    for cfg in results.values():
        all_metrics.update(cfg.get("metrics", {}).keys())
    if not all_metrics:
        lines.append("No metrics available.")
    else:
        priority = [
            "faithfulness", "faithfulness_claim", "answer_relevancy",
            "context_precision", "context_recall", "correctness",
            "precision_at_5", "recall_at_10", "mrr", "ndcg_at_10",
        ]
        ordered = [m for m in priority if m in all_metrics]
        for m in sorted(all_metrics):
            if m not in ordered:
                ordered.append(m)

        # 表头
        header = "| Config | " + " | ".join(ordered) + " | Avg Latency |"
        sep = "|" + "|".join(["---"] * (len(ordered) + 2)) + "|"
        lines.append("## 📊 Metrics")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        for cfg_name, cfg_data in results.items():
        metrics = cfg_data.get("metrics", {})
        row = f"| **{cfg_name}** |"
        for m in ordered:
            val = metrics.get(m)
            row += f" {val:.4f} |" if val is not None else " — |"
        svc_lat = cfg_data.get("avg_service_latency", cfg_data.get("avg_latency", 0))
        row += f" {svc_lat:.2f}s |"
        lines.append(row)

        lines.append("")

    # 延迟表
    timing_cols = [
        ("embedding_ms", "Embedding"),
        ("dense_search_ms", "Dense Search"),
        ("sparse_search_ms", "Sparse Search"),
        ("reranking_ms", "Reranking"),
        ("generation_ms", "Generation"),
        ("scoring_ms", "Scoring"),
    ]
    has_timing = any(
        cfg_data.get("per_component_timing") for cfg_data in results.values()
    )
    if has_timing:
        lines.append("## ⏱️ Latency Breakdown (ms)")
        lines.append("")
        header = "| Config | " + " | ".join(l for _, l in timing_cols) + " | Total |"
        sep = "|" + "|".join(["---"] * (len(timing_cols) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for cfg_name, cfg_data in results.items():
            timing = cfg_data.get("per_component_timing", {})
            row = f"| **{cfg_name}** |"
            total = 0.0
            for key, _ in timing_cols:
                val = timing.get(key, 0) if isinstance(timing, dict) else 0
                row += f" {val:.0f} |"
                total += val
            row += f" **{total:.0f}** |"
            lines.append(row)
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by Enterprise RAG Evaluation Framework at {now}*")
    lines.append("")

    content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Markdown report saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def generate_reports_from_dir(
    results_dir: str,
    output_dir: str | None = None,
    format: str = "both",
) -> dict[str, str]:
    """从结果目录加载数据后生成报告。

    Args:
        results_dir: 包含 eval 结果 JSON 的目录
        output_dir: 输出目录，默认与 results_dir 相同
        format: 'html', 'markdown', 或 'both'

    Returns:
        {"html": path, "markdown": path} (可能只含选中的格式)
    """
    results = _load_results_from_dir(results_dir)
    if not results:
        raise ValueError(f"No eval results found in {results_dir}")

    output_dir = Path(output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    if format in ("html", "both"):
        html_path = str(output_dir / "report.html")
        generate_html_report(results, html_path)
        outputs["html"] = html_path

    if format in ("markdown", "both"):
        md_path = str(output_dir / "report.md")
        generate_markdown_report(results, md_path)
        outputs["markdown"] = md_path

    return outputs
