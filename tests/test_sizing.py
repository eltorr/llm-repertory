"""The sizing model must reproduce what was measured and refuse plans that cannot work."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engines.exl3_offload import sizing  # noqa: E402

GB = 1e9
GIB = 1024 ** 3


class SizingTests(unittest.TestCase):
    def test_matches_rtx_5090_measurements(self):
        # full mode on the 5090: ~28 GB for the model; RAM_GB 26 (3.05) / ~40 (4.05) held every expert;
        # VRAM outside the cache measured 9.0 / 9.44 GB at 262K (research/vram-scaling)
        p = sizing.plan("3.05", 28 * GB, 262144)
        self.assertTrue(p["ok"] and p["staging"])
        self.assertEqual(p["chunk"], 24576)
        self.assertAlmostEqual(p["fixed"] / GB, 9.0, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 26.6, delta=1.0)
        p = sizing.plan("4.05", 28 * GB, 262144)
        self.assertAlmostEqual(p["fixed"] / GB, 9.44, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 41.0, delta=1.5)

    def test_reserve_is_the_one_the_engine_keeps(self):
        # from 16 GB the engine keeps 1.7 GiB, set explicitly; 4.05 on the whole 5090, budget
        # 29.13 GB, loaded 7,264 slots (research/live-resize), as the plan now says
        self.assertEqual(sizing.plan("4.05", 29126885376.0, 262144)["slots"], 7264)
        lean = sizing.plan("4.05", 16 * GB, 262144)
        self.assertEqual(sizing.engine_env(lean)["GLM53_EC_RESERVE_GB"], "1.70")
        self.assertEqual(sizing.engine_env(lean)["GLM53_EC_ELASTIC_GB"], "6.0")
        self.assertEqual(lean["reserve"], sizing.ENGINE_RESERVE_GIB * GIB)
        self.assertEqual(lean["slots"], 1920)       # not 2,272: the engine had 1,824 after a live shrink
        # below 16 GB the engine gets the plan's own reserve
        small = sizing.plan("4.05", 14.5 * GB, 262144)
        self.assertAlmostEqual(float(sizing.engine_env(small)["GLM53_EC_RESERVE_GB"]) * GIB, small["reserve"],
                               delta=0.01 * GIB)

    def test_staging_holds_a_whole_layer(self):
        self.assertEqual(sizing.stage_bytes("4.05"), 2 * 512 * 2_457_600)
        self.assertGreater(512 * 2_457_600 / GIB, 1.1)        # why the 1.1 GiB default overflowed at 4.05

    def test_small_cards_get_smaller_chunks(self):
        big = sizing.plan("3.05", 22 * GB, 65536)
        small = sizing.plan("3.05", 11 * GB, 65536)
        self.assertTrue(big["ok"] and small["ok"])
        self.assertLess(small["chunk"], big["chunk"])

    def test_refuses_what_cannot_fit(self):
        p = sizing.plan("3.05", 6 * GB, 16384)
        self.assertFalse(p["ok"])
        self.assertIn("not enough", p["reason"])
        self.assertEqual(sizing.max_context("3.05", 6 * GB), 0)

    def test_8gb_class_runs_staging_free_without_a_desktop(self):
        # an 8 GB card not driving a display leaves ~8 GB: no room for staging, but the staging-free path fits at 16K
        p = sizing.plan("3.05", 8 * GB, 16384)
        self.assertTrue(p["ok"])
        self.assertFalse(p["staging"])
        self.assertGreaterEqual(p["reserve"], 0.7 * GB)
        # driving a desktop as well (~7 GB left), long prompts overran it: refuse
        self.assertFalse(sizing.plan("3.05", 7 * GB, 16384)["ok"])

    def test_baseline_presets(self):
        def b(vram, ram, display=True, platform="linux"):
            return sizing.baseline(sizing.card_bytes(vram), sizing.ram_bytes(ram), display, platform)
        # this PC: RTX 5090, 64 GB, Windows: what it runs today
        p = b(32, 64, platform="windows-wsl")
        self.assertEqual((p["quant"], p["context"]), ("4.05", 262144))
        self.assertTrue(p["measured"])
        # 24 GB: 4.05 needs the RAM to hold the rest of its experts; 3.05 below that
        self.assertEqual(b(24, 128)["quant"], "4.05")
        self.assertEqual(b(24, 32)["quant"], "3.05")
        self.assertEqual(b(16, 128)["quant"], "3.05")         # 4.05 prefill drops to 4K chunks on 16 GB
        # small cards: 64K staging-free on 12 GB, 16K staging-free on 8 GB only without a display
        p = b(12, 64)
        self.assertEqual((p["context"], p["staging"]), (65536, False))
        p = b(8, 64, display=False)
        self.assertTrue(p["ok"])
        self.assertEqual((p["context"], p["staging"]), (16384, False))
        self.assertFalse(b(8, 64)["ok"])
        # RAM: 16 GB works on Linux with smaller thresholds, not on Windows
        p = b(16, 16)
        self.assertTrue(p["ok"])
        self.assertLess(p["settings"]["ram_reserve_gb"], 6)
        self.assertFalse(b(16, 16, platform="windows-wsl")["ok"])
        # a measurement only labels a preset with a similar RAM cache
        self.assertFalse(b(16, 32)["measured"])

    def test_short_prompts_skip_staging(self):
        staged = sizing.plan("4.05", 28.5e9, 262144)
        self.assertTrue(staged["staging"])
        self.assertEqual(sizing.engine_env(staged)["GLM53_EC_STAGE_MIN"], "16384")   # 1,638 tokens x 10 experts
        free = sizing.plan("3.05", 10.5e9, 131072)
        self.assertFalse(free["staging"])
        self.assertNotIn("GLM53_EC_STAGE_MIN", sizing.engine_env(free))

    def test_ram_cache_floor_and_caps(self):
        policy = {"lean_ram_gb": 8, "ram_reserve_gb": 6, "ram_lean_gb": 6}
        need = 34 * GIB
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB), 34)
        # short of RAM: never below lean's cache (whether lean fits is the pressure policy's call)
        self.assertEqual(sizing.ram_cache_gib(need, 20 * GIB, policy, 8 * GB), 8)
        # a cap above the floor limits the cache; one below it (max_ram_cache_gib) is honoured, not raised to 8
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB, cap_gib=20), 20)
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB, cap_gib=4), 4)

    def test_full_mode_headroom(self):
        self.assertAlmostEqual(sizing.headroom_bytes(32.6e9), 2.4e9)      # ran stably with ~2.1 GB free
        self.assertAlmostEqual(sizing.headroom_bytes(16.3e9), 1.63e9)
        self.assertAlmostEqual(sizing.headroom_bytes(8.5e9, display=False), 0.85e9)

    def test_doc_tables_are_generated(self):
        root = Path(__file__).resolve().parents[1]
        for doc, name, body in (("README.md", "presets", sizing.presets_table()),
                                ("README.md", "measured-summary", sizing.measured_summary()),
                                ("docs/measured.md", "measured", sizing.measured_table("../"))):
            self.assertIn(f"<!-- {name}:begin -->\n{body}\n<!-- {name}:end -->",
                          (root / doc).read_text(encoding="utf-8"),
                          f"{doc}'s {name} table is stale: "
                          f"python3 engines/exl3_offload/sizing.py presets --write {doc}")

    def test_budgeted_runs_pin_the_cache(self):
        p = sizing.plan("3.05", 16 * GB, 262144)
        self.assertIn("GLM53_EC_MAX_SLOTS", sizing.engine_env(p, cap_slots=True))
        self.assertNotIn("GLM53_EC_MAX_SLOTS", sizing.engine_env(sizing.plan("3.05", 28 * GB), cap_slots=False))


if __name__ == "__main__":
    unittest.main()
