"""测试 load_eval_dataset / save_eval_dataset roundtrip。"""

import json
from pathlib import Path
from src.evaluation.dataset import EvalSample, load_eval_dataset, save_eval_dataset


class TestDatasetRoundtrip:
    def test_save_then_load(self, tmp_path):
        samples = [
            EvalSample(
                question="q1",
                ground_truth="a1",
                relevance_labels={"c1": 2},
                expected_sources=["s1.md"],
                difficulty="simple",
            ),
            EvalSample(
                question="q2",
                ground_truth="a2",
                relevance_labels={"c2": 1},
                difficulty="complex",
            ),
        ]
        p = tmp_path / "test.json"
        save_eval_dataset(samples, str(p))

        loaded = load_eval_dataset(str(p))
        assert len(loaded) == 2
        assert loaded[0].question == "q1"
        assert loaded[0].relevance_labels == {"c1": 2}
        assert loaded[0].difficulty == "simple"
        # answer 字段 load 时不回填（有损，符合预期）
        assert loaded[0].answer == ""

    def test_load_nonexistent(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_eval_dataset("/nonexistent/path.json")
