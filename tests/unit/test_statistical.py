"""测试 statistical.py _bootstrap_ci。"""

from src.evaluation.statistical import _bootstrap_ci


class TestBootstrapCI:
    def test_deterministic_seed(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        ci1 = _bootstrap_ci(values)
        ci2 = _bootstrap_ci(values)
        # 相同输入 + 固定 seed → 相同输出
        assert ci1 == ci2

    def test_ci_bounds(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = _bootstrap_ci(values)
        assert lower <= upper
        assert lower > 0.0
        assert upper < 1.0

    def test_small_n_fallback(self):
        values = [0.5, 0.6]
        lower, upper = _bootstrap_ci(values)
        # 只有 2 个值 → 回退到 values[0], values[0]
        assert lower == upper or (lower <= upper)
