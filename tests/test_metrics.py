"""The dashboard's request figures (repertory/service.py Service.record_metrics): prompt speed counts only the prompt
tokens the model processed, not the ones the prefix cache served."""
import collections
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repertory import service  # noqa: E402


def recorded(usage, ttft=2.0, decode_s=10.0):
    svc = object.__new__(service.Service)
    svc.metrics = collections.deque(maxlen=10)
    svc.be = types.SimpleNamespace(tier="full", budget=28.7e9)
    stats = types.SimpleNamespace(status=200, usage=lambda: usage, t_first=100.0 + ttft)
    svc.record_metrics(stats, 99.0, 100.0, 100.0 + ttft + decode_s, False)
    return svc.metrics[-1]


class PromptSpeedTests(unittest.TestCase):
    def test_tokens_from_the_prefix_cache_dont_count_as_processed(self):
        m = recorded({"prompt_tokens": 50_000, "completion_tokens": 400,
                      "prompt_tokens_details": {"cached_tokens": 48_000}})
        self.assertEqual((m["prompt"], m["cached"], m["prefill_tps"]), (50_000, 48_000, 1_000))

    def test_responses_usage_names(self):
        m = recorded({"input_tokens": 10_000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 0}})
        self.assertEqual((m["prompt"], m["cached"], m["prefill_tps"], m["completion"]), (10_000, 0, 5_000, 50))

    def test_almost_all_cached_is_no_prefill_measurement(self):
        m = recorded({"prompt_tokens": 9_000, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 8_900}})
        self.assertIsNone(m["prefill_tps"])                     # 100 new tokens: too few to time

    def test_a_cached_figure_out_of_range_is_ignored(self):
        m = recorded({"prompt_tokens": 4_000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": -5}})
        self.assertEqual((m["cached"], m["prefill_tps"]), (0, 2_000))


if __name__ == "__main__":
    unittest.main()
