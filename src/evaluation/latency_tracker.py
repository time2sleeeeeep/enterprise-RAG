# 分组件延迟追踪模块：提供 LatencyTracker 上下文管理器和 ComponentTiming 数据类。
# 在检索管道的各个环节埋点记录耗时，用于诊断性能瓶颈。

import time
import statistics
from dataclasses import dataclass, field


@dataclass
class ComponentTiming:
    """各组件在一次评估中的累计耗时（毫秒）。"""
    embedding_ms: float = 0.0
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    scoring_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "embedding_ms": self.embedding_ms,
            "dense_search_ms": self.dense_search_ms,
            "sparse_search_ms": self.sparse_search_ms,
            "reranking_ms": self.reranking_ms,
            "generation_ms": self.generation_ms,
            "scoring_ms": self.scoring_ms,
        }

    @property
    def total_ms(self) -> float:
        return sum(self.as_dict().values())


class LatencyTracker:
    """轻量级延迟追踪器，通过 start()/record() 记录各组件耗时。

    用法：
        tracker = LatencyTracker()
        tracker.start()
        do_embedding()
        tracker.record("embedding_ms")
        tracker.start()
        do_search()
        tracker.record("dense_search_ms")
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = {
            "embedding_ms": [],
            "dense_search_ms": [],
            "sparse_search_ms": [],
            "reranking_ms": [],
            "generation_ms": [],
            "scoring_ms": [],
        }
        self._start: float = 0.0

    def start(self) -> None:
        """标记开始计时（使用 perf_counter 纳秒精度转毫秒）。"""
        self._start = time.perf_counter()

    def record(self, component: str) -> float:
        """记录从上次 start() 到现在的耗时（毫秒），追加到对应组件列表，返回该次耗时。"""
        elapsed = (time.perf_counter() - self._start) * 1000
        if component in self._timings:
            self._timings[component].append(elapsed)
        return elapsed

    def to_component_timing(self) -> ComponentTiming:
        """汇总各组件所有样本的累计耗时，返回 ComponentTiming。"""
        return ComponentTiming(
            embedding_ms=sum(self._timings["embedding_ms"]),
            dense_search_ms=sum(self._timings["dense_search_ms"]),
            sparse_search_ms=sum(self._timings["sparse_search_ms"]),
            reranking_ms=sum(self._timings["reranking_ms"]),
            generation_ms=sum(self._timings["generation_ms"]),
            scoring_ms=sum(self._timings["scoring_ms"]),
        )

    def summary(self) -> dict[str, dict[str, float]]:
        """返回每个组件的统计摘要（mean, median, p95, p99, total）。"""
        result = {}
        for name, values in self._timings.items():
            if not values:
                result[name] = {"mean": 0, "median": 0, "p95": 0, "p99": 0, "total": 0, "count": 0}
                continue
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            result[name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[0],
                "p99": sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[0],
                "total": sum(values),
                "count": n,
            }
        return result

    def reset(self) -> None:
        """重置所有计时数据。"""
        for key in self._timings:
            self._timings[key] = []
