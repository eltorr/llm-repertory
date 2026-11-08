"""The results log and the benchmark page (repertory/results.py)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from repertory import results

RUN = {"label": "4.05-card-256k", "quant": "4.05", "usable_gb": 28.9, "card_total_gb": 34.19, "context": 262144,
       "nostage": False, "mtp": 0, "cpu_lane": None, "prefetch": False, "workload": "standard",
       "protocol": "repertory-1", "date": "2026-11-07 19:20", "method": "uncapped card", "finished": True,
       "host": {"gpu": "NVIDIA GeForce RTX 5090", "ram_total_gb": 68.3, "platform": "windows-wsl"},
       "stack": {"torch": "2.15.0+cu132", "cuda": "13.2", "driver": "617.14", "commit": "abcdef0"}, "ram_gib": 35,
       "decode_tps_median": 60.87, "short_ttft_median_s": 0.31,
       "prefill": {"1024": {"median": 2100.0}, "65536": {"median": 5050.0}},
       "long": {"prompt_tokens": 98304, "prefill_tps": 4801.5}}

H2H = {"label": "glm-2.05", "model": "glm-5.3-flash", "date": "2026-11-08 14:00", "limit_s": 1200,
       "setup": {"quant": "glm-2.05", "tier": "full", "context": 262144, "ram_cache_gib": 37, "vram_budget_gb": 28.9,
                 "gpu": "NVIDIA GeForce RTX 5090", "card_gb": 34.19, "ram_gb": 68.3, "platform": "windows-wsl"},
       "tasks": [{"id": "calc", "ok": True, "completion_tokens": 1700, "seconds": 222.0, "timed_out": False},
                 {"id": "paths", "ok": False, "completion_tokens": 0, "seconds": 1200.0, "timed_out": True},
                 {"id": "sched", "ok": False, "completion_tokens": 1704, "seconds": 204.0, "timed_out": False}]}


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "results"

    def save(self, name, data):
        p = self.tmp / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_a_run_is_logged_with_its_gpu_preset_and_model(self):
        r = results.speed_row(RUN)
        self.assertEqual((r["gpu"], r["card"], r["capped"], r["ram_preset"], r["platform"]),
                         ("RTX 5090", "32 GB", False, 64, "windows-wsl"))
        self.assertEqual((r["model"], r["bpw"], r["context"], r["ram_cache_gib"]), ("Qwen3.8-Flash-Next", 4.05,
                                                                                    262144, 35))
        self.assertEqual((r["decode_tps"], r["prefill_tps"], r["prefill_tokens"]), (60.87, 4801.5, 98304))
        self.assertEqual(r["stack"]["commit"], "abcdef0")
        self.assertNotIn("host", r)

    def test_a_capped_run_is_the_smaller_cards_preset(self):
        r = results.speed_row(dict(RUN, usable_gb=15.9, method="engine cap (limit + guard)", long={}))
        self.assertEqual((r["card"], r["capped"]), ("16 GB", True))
        self.assertEqual((r["prefill_tps"], r["prefill_tokens"]), (5050.0, 65536))   # the longest document then

    def test_runs_that_dont_count_are_not_logged(self):
        for bad in ({"error": "does not fit: VRAM"}, {"error": "cancelled"}, {"interrupted": True},
                    {"finished": False}):
            self.assertIsNone(results.speed_row(dict(RUN, **bad)), bad)

    def test_engine_options_are_named(self):
        r = results.speed_row(dict(RUN, mtp=2, cpu_lane="", prefetch=True))
        self.assertEqual(r["features"], ["mtp=2", "cpu", "prefetch"])

    def test_a_model_no_longer_in_the_catalog_keeps_its_quant(self):
        r = results.speed_row(dict(RUN, quant="9.99"))
        self.assertEqual((r["model"], r["bpw"]), ("9.99", None))

    def test_ram_presets(self):
        self.assertEqual([results.ram_preset(g) for g in (17.1, 34.2, 68.3, 137.0)], [16, 32, 64, 128])
        self.assertIsNone(results.ram_preset(None))

    def test_a_head_to_head_is_logged_with_its_setup(self):
        r = results.quality_row(H2H)
        self.assertEqual((r["model"], r["bpw"], r["card"], r["ram_preset"]), ("GLM-5.3-Flash", 2.05, "32 GB", 64))
        self.assertEqual((r["passed"], r["total"], r["limit_min"]), (1, 3, 20))
        self.assertEqual(r["tasks"][1], {"id": "paths", "ok": False, "tokens": 0, "minutes": 20.0, "cut_off": True})
        self.assertIsNone(results.quality_row({k: v for k, v in H2H.items() if k != "setup"}))

    def test_adding_a_run_again_replaces_its_line(self):
        run, h2h = self.save("run.json", RUN), self.save("h2h.json", H2H)
        said = results.add([run, h2h], self.log)
        self.assertTrue(said[0].startswith("added") and said[1].startswith("added"), said)
        said = results.add([self.save("run.json", dict(RUN, decode_tps_median=61.0))], self.log)
        self.assertTrue(said[0].startswith("replaced"), said)
        speed = results.read(results.SPEED, self.log)
        self.assertEqual([r["decode_tps"] for r in speed], [61.0])
        self.assertEqual(len(results.read(results.QUALITY, self.log)), 1)

    def test_a_bench_folder_logs_its_completed_runs_only(self):
        d = self.tmp / "20261107-191719"
        d.mkdir()
        (d / "_plan.json").write_text(json.dumps({"runs": []}))
        (d / "a.json").write_text(json.dumps(RUN))
        (d / "b.json").write_text(json.dumps(dict(RUN, label="3.05-card-256k", quant="3.05", error="cancelled")))
        said = results.add([d], self.log)
        self.assertEqual(sum(s.startswith("added") for s in said), 1, said)
        self.assertEqual(sum(s.startswith("skipped") for s in said), 1, said)

    def test_the_page_groups_speed_by_machine_and_puts_bigger_cards_first(self):
        small = dict(RUN, label="4.05-16-256k", usable_gb=15.9, method="engine cap (limit + guard)")
        other = dict(RUN, host={"gpu": "NVIDIA GeForce RTX 4090", "ram_total_gb": 34.2, "platform": "linux"},
                     card_total_gb=25.76, usable_gb=21.4)
        results.add([self.save("a.json", small), self.save("b.json", RUN), self.save("c.json", other),
                     self.save("h.json", H2H)], self.log)
        page = results.render(self.log)
        rtx4090 = page.index("### RTX 4090 (24 GB), 32 GB RAM, Linux")
        rtx5090 = page.index("### RTX 5090 (32 GB), 64 GB RAM, Windows (WSL2)")
        self.assertLess(rtx4090, rtx5090)
        section = page[rtx5090:]
        self.assertLess(section.index("| 32 GB card |"), section.index("| 16 GB card (capped) |"))
        self.assertIn("| GLM-5.3-Flash | 2.05 bpw | 32 GB card | 1/3 | pass (3.7 min) | fail (3.4 min) | - | "
                      "cut off (20 min) |", page)
        self.assertIn("4,802 tok/s (98K)", page)

    def test_the_page_is_written_between_its_markers(self):
        results.add([self.save("b.json", RUN)], self.log)
        page = self.tmp / "benchmarks.md"
        page.write_text("# B\n\n<!-- results:begin -->\n<!-- results:end -->\n\nafter\n", encoding="utf-8")
        results.write_page(page, self.log)                                      # a new page: nothing between
        self.assertIn("## Speed, by GPU", page.read_text(encoding="utf-8"))
        page.write_text("# B\n\n<!-- results:begin -->\nold\n<!-- results:end -->\n\nafter\n", encoding="utf-8")
        results.write_page(page, self.log)
        text = page.read_text(encoding="utf-8")
        self.assertNotIn("old", text)
        self.assertTrue(text.endswith("<!-- results:end -->\n\nafter\n"))
        self.assertIn("## Speed, by GPU", text)
        page.write_text("no markers", encoding="utf-8")
        with self.assertRaises(SystemExit):
            results.write_page(page, self.log)

    def test_the_log_has_no_host_names_prompts_or_answers(self):
        h2h = copy.deepcopy(H2H)
        h2h["tasks"][0].update(answer="secret answer", reasoning="secret reasoning")
        results.add([self.save("h.json", h2h), self.save("b.json", dict(RUN, hostname="my-pc"))], self.log)
        text = "".join((self.log / n).read_text() for n in (results.SPEED, results.QUALITY))
        for s in ("secret", "my-pc", "prompt\""):
            self.assertNotIn(s, text)


if __name__ == "__main__":
    unittest.main()
